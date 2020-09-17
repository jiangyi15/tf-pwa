#include "TCollection.h"

const char *get_name(TTree *t, int idx, bool using_count = false) {
    int n_branch = t->GetNbranches();
    auto it_branch = t->GetListOfBranches();
    std::ostringstream head_os;
    head_os << "MC_" << idx;
    std::string head = head_os.str();
    int n_head = head.length();
    int count = 0;
    for (auto br : (*it_branch)) {
        const char *name = ((TBranch *)br)->GetName();
        if (using_count) {
            if (count == idx) {
                return name;
            }
            count++;
        } else {
            if (head.compare(0, n_head, name, 0, n_head) == 0) {
                return name;
            }
        }
    }
    return "";
}

TH1D *fill_hist(TString name, TTree *t, TString var, TString weight,
                double v_min, double v_max, int bins) {
    TH1D *histo = new TH1D(name, name, bins, v_min, v_max);
    double val, val_w;
    bool have_weight = weight.Length() >= 1;
    t->SetBranchAddress(var, &val);
    if (have_weight)
        t->SetBranchAddress(weight, &val_w);
    for (int i = 0; i < t->GetEntries(); i++) {
        t->GetEntry(i);
        if (have_weight)
            histo->Fill(val, val_w);
        else
            histo->Fill(val);
    }
    return histo;
}

const double COLOR_TABLE[] = {
    920,
    632,
    416,
    /* kBlue=600, */ 400,
    616,
    432,
    800,
    820,
    840,
    860,
    880,
    900,
};

void draw_root(TString var = "m_BC") {

    double v_min = 3.8;
    double v_max = 4.82;
    int n_bins = 50;
    int bin_scale = 3;

    TFile file("variables.root");
    TTree *data = (TTree *)file.Get("data");
    TTree *fitted = (TTree *)file.Get("fitted");
    TTree *bg = (TTree *)file.Get("sideband");

    gStyle->SetOptStat(0);

    TCanvas *histosPrint = new TCanvas();
    histosPrint->cd();
    TString var_bg = var + "_sideband";
    TString var_mc = var + "_MC";

    auto hist_data = fill_hist("all data", data, var, "", v_min, v_max, n_bins);
    hist_data->SetLineWidth(2);
    hist_data->Draw("E");

    auto hist_bg =
        fill_hist("bg", bg, var_bg, "sideband_weights", v_min, v_max, n_bins);
    hist_bg->SetLineColor(3);
    hist_bg->SetFillColor(3);
    hist_bg->DrawCopy("HISTsame");
    auto hist_fit = fill_hist("totalfit", fitted, var_mc, "MC_total_fit", v_min,
                              v_max, n_bins);

    hist_fit->Add(hist_bg);
    hist_data->SetLineWidth(2);
    hist_fit->Draw("HISTsame");

    auto leg = new TLegend(0.7, 0.6, 0.95, 0.95);
    leg->AddEntry(hist_data, "Data", "lep");
    leg->AddEntry(hist_bg, "Background", "f");
    leg->AddEntry(hist_fit, "Total fit", "lep");

    int i = 0;
    while (1) {
        auto name = get_name(fitted, i);
        if (strlen(name) <= 1)
            break;
        auto histo = fill_hist(name, fitted, var_mc, name, v_min, v_max,
                               n_bins * bin_scale);
        std::cout << name << std::endl;
        histo->SetLineColor(COLOR_TABLE[i % 13]);
        histo->SetMarkerStyle(i / 13);
        histo->SetLineWidth(2);
        leg->AddEntry(histo, name, "lep");
        histo->Draw("LHISTsame");
        i++;
    }

    hist_data->Draw("Esame");
    leg->Draw();
    histosPrint->SaveAs("c1_" + var + ".png");
    histosPrint->SaveAs("c1_" + var + ".C");
}
