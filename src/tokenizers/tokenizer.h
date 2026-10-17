#ifndef __SD_TOKENIZERS_TOKENIZER_H__
#define __SD_TOKENIZERS_TOKENIZER_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using on_new_token_cb_t = std::function<bool(std::string&, std::vector<int32_t>&)>;

class Tokenizer {
    friend class TokenStreamDecoder;

protected:
    std::vector<std::string> special_tokens;
    bool add_bos_token          = false;
    bool add_eos_token          = false;
    bool pad_left               = false;
    bool normalize_before_split = false;
    std::string end_of_word_suffix;

    virtual std::string decode_token(int token_id) const = 0;
    virtual std::string normalize(const std::string& text) const;

public:
    std::string UNK_TOKEN;
    std::string BOS_TOKEN;
    std::string EOS_TOKEN;
    std::string PAD_TOKEN;
    int UNK_TOKEN_ID = 0;
    int BOS_TOKEN_ID = 0;
    int EOS_TOKEN_ID = 0;
    int PAD_TOKEN_ID = 0;

    virtual ~Tokenizer() = default;

    virtual void add_special_token(const std::string& token);
    bool is_special_token(const std::string& token) const;
    // An empty output may be valid; failures return false and clear the output.
    virtual bool encode(const std::string& text, std::vector<int>& tokens, on_new_token_cb_t on_new_token_cb = nullptr, std::string* error = nullptr) = 0;
    bool tokenize(const std::string& text,
                  std::vector<int>& tokens,
                  on_new_token_cb_t on_new_token_cb = nullptr,
                  bool padding                      = false,
                  size_t min_length                 = 0,
                  size_t max_length                 = 100000000,
                  bool allow_overflow_expand        = false,
                  std::string* error                = nullptr);
    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>* weights,
                    std::vector<float>* mask,
                    size_t min_length          = 0,
                    size_t max_length          = 100000000,
                    bool allow_overflow_expand = false);
    virtual bool decode(const std::vector<int>& tokens, std::string& text, std::string* error = nullptr) const;
};

// Decodes generated tokens one at a time. Unlike Tokenizer::decode, the text is kept verbatim.
// Bytes of a character split across tokens are held back until it is complete, so every returned
// chunk is valid UTF-8; invalid bytes become U+FFFD.
class TokenStreamDecoder {
private:
    const Tokenizer& tokenizer_;
    bool skip_special_tokens_;
    std::string pending_;

    std::string take_text(bool end_of_stream);

public:
    explicit TokenStreamDecoder(const Tokenizer& tokenizer, bool skip_special_tokens = true);

    // Returns the text completed by this token; it may be empty.
    std::string push(int token_id);
    // Returns what is still held back at the end of the stream and resets the decoder.
    std::string flush();
};

#endif  // __SD_TOKENIZERS_TOKENIZER_H__
