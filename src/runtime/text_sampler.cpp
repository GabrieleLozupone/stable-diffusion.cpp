#include "runtime/text_sampler.h"

#include <algorithm>
#include <cmath>

namespace sd::text {

    Sampler::Sampler(const SamplingParams& params, uint64_t seed)
        : params_(params), rng_(seed) {
    }

    double Sampler::next_uniform() {
        return static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0);
    }

    void Sampler::accept(int32_t token) {
        if (token < 0) {
            return;
        }
        if (static_cast<size_t>(token) >= seen_.size()) {
            seen_.resize(static_cast<size_t>(token) + 1, 0);
        }
        seen_[token] = 1;
    }

    float Sampler::penalized(const float* logits, int32_t token) const {
        float logit = logits[token];
        if (params_.presence_penalty != 0.f && static_cast<size_t>(token) < seen_.size() && seen_[token]) {
            logit -= params_.presence_penalty;
        }
        return logit;
    }

    std::vector<std::pair<int32_t, double>> Sampler::candidates(const float* logits, int64_t n_vocab) const {
        // Ties keep the lower id first, as argmax does.
        auto by_logit = [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        };
        std::vector<std::pair<int32_t, float>> cand;
        if (params_.top_k > 0 && params_.top_k < n_vocab) {
            // One pass with a min-heap of the k best: the vocabulary is ~150k tokens and k is small.
            const size_t k = static_cast<size_t>(params_.top_k);
            cand.reserve(k + 1);
            for (int64_t i = 0; i < n_vocab; i++) {
                std::pair<int32_t, float> c(static_cast<int32_t>(i), penalized(logits, static_cast<int32_t>(i)));
                if (cand.size() < k) {
                    cand.push_back(c);
                    std::push_heap(cand.begin(), cand.end(), by_logit);
                } else if (by_logit(c, cand.front())) {
                    std::pop_heap(cand.begin(), cand.end(), by_logit);
                    cand.back() = c;
                    std::push_heap(cand.begin(), cand.end(), by_logit);
                }
            }
            std::sort_heap(cand.begin(), cand.end(), by_logit);
        } else {
            cand.reserve(static_cast<size_t>(n_vocab));
            for (int64_t i = 0; i < n_vocab; i++) {
                cand.emplace_back(static_cast<int32_t>(i), penalized(logits, static_cast<int32_t>(i)));
            }
            std::sort(cand.begin(), cand.end(), by_logit);
        }

        const double inv_t = params_.temperature > 0.f ? 1.0 / params_.temperature : 1.0;
        const double max_l = cand.front().second;
        std::vector<std::pair<int32_t, double>> probs;
        probs.reserve(cand.size());
        double sum = 0.0;
        for (const auto& c : cand) {
            const double p = std::exp((c.second - max_l) * inv_t);
            probs.emplace_back(c.first, p);
            sum += p;
        }
        for (auto& p : probs) {
            p.second /= sum;
        }

        // Smallest prefix whose probability reaches top_p (at least one token).
        if (params_.top_p < 1.f) {
            double cum  = 0.0;
            size_t keep = probs.size();
            for (size_t i = 0; i < probs.size(); i++) {
                cum += probs[i].second;
                if (cum >= params_.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            probs.resize(keep);
            double kept = 0.0;
            for (const auto& p : probs) {
                kept += p.second;
            }
            for (auto& p : probs) {
                p.second /= kept;
            }
        }
        return probs;
    }

    int32_t Sampler::sample(const float* logits, int64_t n_vocab) {
        int32_t token = 0;
        if (greedy() && params_.presence_penalty == 0.f) {
            token = static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits);
        } else if (greedy()) {
            float best = penalized(logits, 0);
            for (int64_t i = 1; i < n_vocab; i++) {
                const float l = penalized(logits, static_cast<int32_t>(i));
                if (l > best) {
                    best  = l;
                    token = static_cast<int32_t>(i);
                }
            }
        } else {
            const auto probs = candidates(logits, n_vocab);
            const double u   = next_uniform();
            double cum       = 0.0;
            token            = probs.back().first;
            for (const auto& p : probs) {
                cum += p.second;
                if (u < cum) {
                    token = p.first;
                    break;
                }
            }
        }
        accept(token);
        return token;
    }

}
