#include "generation.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

#include "core/util.h"
#include "diffusion_engine.h"
#include "runtime/text_sampler.h"

namespace sd::pipeline {

    namespace {

        // Prompt plus generated tokens; the KV cache costs ~144 KB per token on Qwen3-VL 8B.
        constexpr int64_t TEXT_GEN_MAX_CONTEXT = 16384;

        using text_clock = std::chrono::steady_clock;

        float seconds_since(text_clock::time_point t0) {
            return std::chrono::duration<float>(text_clock::now() - t0).count();
        }

        LLMEmbedder* text_generation_embedder(StableDiffusionGGML* sd) {
            auto embedder = std::dynamic_pointer_cast<LLMEmbedder>(sd->cond_stage_model);
            if (embedder == nullptr || embedder->llm == nullptr || embedder->tokenizer == nullptr) {
                return nullptr;
            }
            // The chat template and the image layout below are Qwen3-VL's; other LLMs are untested.
            const auto& llm = *embedder->llm;
            if (llm.config.arch != LLM::LLMArch::QWEN3_VL || !llm.supports_kv_cache()) {
                return nullptr;
            }
            return embedder.get();
        }

        // Size the vision tower sees: a multiple of the merged patch, scaled down to max_pixels if set.
        void vision_size(int width, int height, int factor, int max_pixels, int& out_w, int& out_h) {
            double scale = 1.0;
            if (max_pixels > 0 && static_cast<double>(width) * height > max_pixels) {
                scale = std::sqrt(static_cast<double>(max_pixels) / (static_cast<double>(width) * height));
                out_w = std::max(factor, static_cast<int>(std::floor(width * scale / factor)) * factor);
                out_h = std::max(factor, static_cast<int>(std::floor(height * scale / factor)) * factor);
                return;
            }
            out_w = std::max(factor, static_cast<int>(std::lround(static_cast<double>(width) / factor)) * factor);
            out_h = std::max(factor, static_cast<int>(std::lround(static_cast<double>(height) / factor)) * factor);
        }

        // RGB in [-1, 1], alpha composited on white like the Qwen-Image 2.1 conditioner.
        sd::Tensor<float> vision_input(const sd_image_t& image, int width, int height) {
            sd::Tensor<float> src = sd_image_to_tensor(image, width, height);
            const int64_t pixels  = static_cast<int64_t>(width) * height;
            const int64_t ch      = src.shape()[2];
            sd::Tensor<float> rgb({width, height, 3, 1});
            for (int64_t p = 0; p < pixels; ++p) {
                const float alpha = ch == 4 ? src[p + 3 * pixels] : 1.f;
                for (int c = 0; c < 3; ++c) {
                    const float v       = src[p + std::min<int64_t>(c, ch - 1) * pixels];
                    rgb[p + c * pixels] = 2.f * (v * alpha + 1.f - alpha) - 1.f;
                }
            }
            return rgb;
        }

        char* copy_string(const std::string& s) {
            char* out = static_cast<char*>(malloc(s.size() + 1));
            if (out != nullptr) {
                memcpy(out, s.c_str(), s.size() + 1);
            }
            return out;
        }

    }

    bool supports_text_generation(StableDiffusionGGML* sd) {
        return sd != nullptr && text_generation_embedder(sd) != nullptr;
    }

    char* generate_text(StableDiffusionGGML* sd,
                        const sd_text_gen_params_t* params,
                        sd_text_cb_t on_piece,
                        void* data,
                        sd_text_gen_result_t* result) {
        sd_text_gen_result_t res = {};
        if (result != nullptr) {
            *result = res;
        }
        LLMEmbedder* embedder = text_generation_embedder(sd);
        if (embedder == nullptr) {
            LOG_ERROR("text generation needs a Qwen3-VL text encoder");
            return nullptr;
        }
        if (params->prompt == nullptr || params->max_tokens <= 0 || params->images_count < 0 ||
            (params->images_count > 0 && params->images == nullptr)) {
            LOG_ERROR("invalid text generation parameters");
            return nullptr;
        }
        LLM::LLMRunner& llm  = *embedder->llm;
        Tokenizer& tokenizer = *embedder->tokenizer;
        if (params->images_count > 0 && !llm.enable_vision) {
            LOG_ERROR("images need the Qwen3-VL vision weights (--llm_vision or a combined encoder)");
            return nullptr;
        }
        sd->reset_cancel_flag();
        auto t_start = text_clock::now();

        std::set<int32_t> stop_tokens;
        for (const char* special : {"<|im_end|>", "<|endoftext|>"}) {
            std::vector<int> id;
            if (!tokenizer.encode(special, id) || id.size() != 1) {
                LOG_ERROR("the tokenizer has no %s token", special);
                return nullptr;
            }
            stop_tokens.insert(id[0]);
        }

        // Chat template with one <|image_pad|> per image, which the LLM replaces with all of the image's
        // embeddings; indices count the expanded sequence. Image token counts are known from the sizes
        // before running the vision tower.
        const int patch  = llm.config.vision.patch_size;
        const int factor = patch * llm.config.vision.spatial_merge_size;
        std::string prompt;
        if (params->system_prompt != nullptr && params->system_prompt[0] != '\0') {
            prompt += "<|im_start|>system\n" + std::string(params->system_prompt) + "<|im_end|>\n";
        }
        prompt += "<|im_start|>user\n";
        std::vector<std::pair<int, int>> sizes;
        std::vector<int> embed_index;
        int image_extra_tokens = 0;
        for (int i = 0; i < params->images_count; ++i) {
            const sd_image_t& image = params->images[i];
            if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel == 0 || image.channel > 4) {
                LOG_ERROR("invalid image %d", i);
                return nullptr;
            }
            int w = 0, h = 0;
            vision_size(static_cast<int>(image.width), static_cast<int>(image.height), factor, params->image_max_pixels, w, h);
            if (params->images_count > 1) {
                prompt += (i == 0 ? "" : " ") + std::string("<image") + std::to_string(i + 1) + ">";
            }
            prompt += "<|vision_start|>";
            std::vector<int> prefix;
            if (!tokenizer.encode(prompt, prefix)) {
                return nullptr;
            }
            embed_index.push_back(static_cast<int>(prefix.size()) + image_extra_tokens);
            sizes.emplace_back(w, h);
            image_extra_tokens += (w / factor) * (h / factor) - 1;
            prompt += "<|image_pad|><|vision_end|>";
        }
        prompt += std::string(params->prompt) + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> ids;
        if (!tokenizer.encode(prompt, ids) || ids.empty()) {
            LOG_ERROR("failed to tokenize the prompt");
            return nullptr;
        }
        const int64_t prompt_tokens = static_cast<int64_t>(ids.size()) + image_extra_tokens;
        const int64_t n_ctx         = prompt_tokens + params->max_tokens;
        if (n_ctx > TEXT_GEN_MAX_CONTEXT) {
            LOG_ERROR("prompt (%" PRId64 " tokens) + max_tokens (%d) exceeds the %" PRId64 "-token text generation limit",
                      prompt_tokens, params->max_tokens, TEXT_GEN_MAX_CONTEXT);
            return nullptr;
        }
        res.prompt_tokens = static_cast<int>(prompt_tokens);

        LLM::LLMRunner* runner = sd->prepare_text_generation(n_ctx);
        if (runner == nullptr) {
            return nullptr;
        }
        res.prepare_s = seconds_since(t_start);

        LLM::ImageEmbeds image_embeds;
        LLM::DeepStackImageEmbeds deepstack;
        std::vector<LLM::ImageGrid> grids;
        auto t_image = text_clock::now();
        for (int i = 0; i < params->images_count; ++i) {
            const auto [w, h] = sizes[i];
            auto outputs      = runner->encode_image_outputs(sd->n_threads, vision_input(params->images[i], w, h), false);
            const int count   = (w / factor) * (h / factor);
            if (outputs.size() < 2 || outputs[0].shape()[1] != count) {
                LOG_ERROR("failed to encode image %d", i);
                runner->runner_end();
                return nullptr;
            }
            image_embeds.emplace_back(embed_index[i], std::move(outputs[0]));
            deepstack.resize(outputs.size() - 1);
            for (size_t layer = 1; layer < outputs.size(); ++layer) {
                deepstack[layer - 1].emplace_back(embed_index[i], std::move(outputs[layer]));
            }
            grids.push_back({embed_index[i], count, h / patch, w / patch});
        }
        res.image_s = params->images_count > 0 ? seconds_since(t_image) : 0.f;

        const sd_text_sample_params_t& sp = params->sample_params;
        res.seed                          = sp.seed >= 0 ? sp.seed : static_cast<int64_t>(std::random_device{}() & 0x7fffffff);
        sd::text::SamplingParams sampling;
        sampling.temperature      = sp.temperature;
        sampling.top_p            = sp.top_p;
        sampling.top_k            = sp.top_k;
        sampling.presence_penalty = sp.presence_penalty;
        sd::text::Sampler sampler(sampling, static_cast<uint64_t>(res.seed));

        TokenStreamDecoder stream(tokenizer, true);
        std::string text;
        bool cancelled = false;
        auto emit      = [&](const std::string& piece) {
            if (piece.empty()) {
                return true;
            }
            text += piece;
            return on_piece == nullptr || on_piece(piece.c_str(), data);
        };
        // Decode time runs from the first to the last token, so it leaves out releasing the KV cache.
        auto t_decode    = text_clock::now();
        auto t_last      = t_decode;
        bool first_token = true;
        auto on_token    = [&](int32_t token, const sd::Tensor<float>&) {
            t_last = text_clock::now();
            if (first_token) {
                res.prefill_s = std::chrono::duration<float>(t_last - t_decode).count();
                t_decode      = t_last;
                first_token   = false;
            }
            if (stop_tokens.count(token)) {
                return true;
            }
            if (sd->get_cancel_flag() == SD_CANCEL_ALL || !emit(stream.push(token))) {
                cancelled = true;
                return false;
            }
            return true;
        };
        std::vector<int32_t> tokens;
        const bool ok = runner->generate(sd->n_threads, sd::Tensor<int32_t>::from_vector(ids), image_embeds, deepstack,
                                         grids, params->max_tokens, stop_tokens, tokens, &sampler, on_token);
        res.decode_s  = std::chrono::duration<float>(t_last - t_decode).count();
        if (!ok) {
            LOG_ERROR("text generation failed after %zu tokens", tokens.size());
            return nullptr;
        }
        if (!cancelled && !emit(stream.flush())) {
            cancelled = true;
        }
        const bool stopped   = !tokens.empty() && stop_tokens.count(tokens.back());
        res.generated_tokens = static_cast<int>(tokens.size()) - (stopped ? 1 : 0);
        res.finish_reason    = cancelled ? SD_TEXT_FINISH_CANCELLED : stopped ? SD_TEXT_FINISH_STOP
                                                                              : SD_TEXT_FINISH_LENGTH;
        LOG_INFO("text generation: %d prompt tokens, %d generated in %.2fs (prefill %.2fs, %.1f tok/s), seed %" PRId64,
                 res.prompt_tokens, res.generated_tokens, seconds_since(t_start), res.prefill_s,
                 res.decode_s > 0 ? (tokens.size() - 1) / res.decode_s : 0.0, res.seed);
        if (result != nullptr) {
            *result = res;
        }
        return copy_string(text);
    }

}
