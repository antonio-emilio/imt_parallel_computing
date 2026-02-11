#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>

#include <openssl/sha.h>

#include <omp.h>

#define SERIAL 0
#define PARALLEL 1

#define STRATEGY SERIAL

std::vector<std::string> read_dict(const std::filesystem::path& path, size_t words_to_read)
{
    std::ifstream if_dict(path);
    std::vector<std::string> words(words_to_read);
    size_t w;

    for(w = 0; w < words_to_read; ++w)
    {
        if(!if_dict.good())
            break;
        std::getline(if_dict, words[w]);
    }
    if(w < words_to_read)
    {
        words.resize(w - 1);
        std::cerr << "The dictionary contained less words than requested." << std::endl;
    }
    std::cout << "Loaded " << w  << " words." << std::endl;

    return words;
}

std::string sha512(const std::string& str)
{
    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512((const unsigned char*)str.c_str(), str.size(), hash);
    std::stringstream ss;
    for(int i = 0; i < SHA512_DIGEST_LENGTH; i++)
    {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

std::string look_for_hash(const std::string& hash_to_look_for, const std::vector<std::string>& words)
{
    std::string found_word;
#if STRATEGY == SERIAL
    for(size_t i = 0; i < words.size(); ++i)
    {
        if(sha512(words[i]) == hash_to_look_for)
        {
            found_word = words[i];
            std::cerr << "I have found the word!" << std::endl;
            break;
        }
    }
#elif STRATEGY == PARALLEL
    // TODO: Find the has using all available threads
#else
    std::cerr << "Not implemented." << std::endl;
#endif

    return found_word;
}

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " dictionary_filename number_words_to_load hash_to_look_for" << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path dict_path(argv[1]);
    size_t n_words = std::stoull(argv[2]);
    if(!std::filesystem::is_regular_file(std::filesystem::status(dict_path)))
    {
        std::cerr << "Invalid dictionary file." << std::endl;
        return EXIT_FAILURE;
    }

    std::string hash_to_look_for(argv[3]);

    // Load words
    std::vector<std::string> words = read_dict(dict_path, n_words);

    /*
     * Main work code
     */

    auto start = std::chrono::steady_clock::now();
    std::string word = look_for_hash(hash_to_look_for, words);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    if(word.size() != 0)
        std::cout << "Found the word " << word << " hasing to " << sha512(word) << std::endl;
    else
        std::cout << "Did not find any word hasing to " << hash_to_look_for << std::endl;
    
    std::cout << "Total search time : " << elapsed_seconds.count() << "s" << std::endl;

    return EXIT_SUCCESS;
}

