 
#include <TChain.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// clang-format off
const std::vector<std::string> param_list = {
  "m_BC", "m_BD", "m_CD",
  "cosTheta_BC", "cosTheta_B_BC", "cosTheta_E_D",
  "phi_B_BC", "phi_E_D", 
  "cosTheta_BD", "cosTheta_D_BD", "cosTheta_E_D_BD",
  "phi_BD", "phi_D_BD", "phi_E_D_BD",
  "cosTheta_CD", "cosTheta_D_CD", "cosTheta_E_D_CD",
  "phi_CD", "phi_D_CD", "phi_E_D_CD",
  "cosTheta1", "cosTheta2", "cosTheta3", 
  "phi1", "phi2", "phi3"};
// clang-format on

void save_params(std::ofstream &outfile, std::string datafile,
                 std::string name) {
  TChain *data = new TChain("t_angle");
  data->Add(datafile.c_str());

  double tmp = 0.0;
  data->SetBranchAddress(name.c_str(), &tmp);

  int N = data->GetEntries();

  outfile << "\"" << name << "\": [";
  for (int i = 0; i < N; i++) {
    data->GetEntry(i);
    if (i > 0)
      outfile << ",";
    outfile << tmp << std::endl;
  }
  outfile << "]";
  delete data;
}

void save_root_to_json(std::string name = "data") {
  std::ofstream outfile;
  int n = param_list.size();
  outfile.open("./data/" + name + ".json");
  outfile << "{";
  for (int i = 0; i < n; i++) {
    save_params(outfile, "./data/" + name + "_angle.root", param_list[i]);
    if (i < n - 1)
      outfile << ",";
  }
  outfile << "}";
  outfile.close();
  std::cout << "convert ./data/" + name + "_angle.root to "
            << "./data/" + name + ".json" << std::endl;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    save_root_to_json();
  } else {
    for (int i = 1; i < argc; i++) {
      save_root_to_json(std::string(argv[i]));
    }
  }
  return 0;
}
