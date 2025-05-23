/* Function declarations */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

std::tuple<std::string, std::string> get_options(int argc, char *argv[]);
std::tuple<std::vector<int>, int> read_data(std::string input_filename);

void validate_input(std::vector<int> ks, int m_in);

#endif // PREPROCESSING_H