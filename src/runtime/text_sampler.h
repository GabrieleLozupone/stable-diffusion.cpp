#ifndef __SD_RUNTIME_TEXT_SAMPLER_H__
#define __SD_RUNTIME_TEXT_SAMPLER_H__

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace sd::text {

    struct SamplingParams {
        float temperature      = 0.f;  // <= 0: greedy (argmax)
        float top_p            = 1.f;  // >= 1: off
        int top_k              = 0;    // <= 0: off
        float presence_penalty = 0.f;  // subtracted once from every token already generated
    };

    // Next-token sampler for text generation. Steps, in the order of HF transformers and vLLM:
    // presence penalty, temperature, top-k, top-p, then one draw. The presence penalty only counts
    // generated tokens (vLLM semantics), so the prompt's own words are never penalized.
    // Draws are reproducible across platforms: mt19937_64 output is specified by the standard, and
    // the conversion to [0, 1) is done here instead of std::uniform_real_distribution.
    class Sampler {
    public:
        Sampler(const SamplingParams& params, uint64_t seed);

        bool greedy() const {
            return params_.temperature <= 0.f;
        }

        // Picks the next token from raw logits and records it for the presence penalty.
        int32_t sample(const float* logits, int64_t n_vocab);

        // Records a token generated elsewhere (the presence penalty applies to it from now on).
        void accept(int32_t token);

        // Uniform double in [0, 1) with 53 random bits.
        double next_uniform();

        // Probabilities after penalty, temperature, top-k and top-p, sorted by decreasing
        // probability; what sample() draws from when not greedy. Exposed for tests.
        std::vector<std::pair<int32_t, double>> candidates(const float* logits, int64_t n_vocab) const;

    private:
        float penalized(const float* logits, int32_t token) const;

        SamplingParams params_;
        std::mt19937_64 rng_;
        std::vector<uint8_t> seen_;
    };

}

#endif  // __SD_RUNTIME_TEXT_SAMPLER_H__
