romelli_cbis = c("ro_cbie_index", "ro_cbie_board", "ro_cbie_policy", "ro_cbie_obj", "ro_cbie_lending", "ro_cbie_finindep", "ro_cbie_report")

dom <- c("monetary_dominance", "financial_dominance", "fiscal_dominance")
dom_label <- c("Monetary dominance", "Financial dominance", "Fiscal dominance")
named_dom <- set_names(dom, dom_label)

dom_corp <- c("monetary_dominance", "financial_dominance", "fiscal_dominance", "monetary_fiscal_coordination", "monetary_financial_coordination")
dom_corp_label <- c("Monetary dominance", "Financial dominance", "Fiscal dominance", "Monetary-fiscal coordination", "Monetary-financial coordination")
named_dom_corp <- set_names(dom_corp, dom_corp_label)

audiences <- c("academic", "central_bank", "financial_market", "political")
audiences_labels <- c("Academic", "General Central Banking", "Financial Market", "Political")

named_audiences <- audiences %>%
  set_names(audiences_labels)



convert_dominance_name <- function(varname) {
  return(list("monetary_dominance" = "Monetary dominance",
              "financial_dominance" = "Financial dominance",
              "fiscal_dominance" = "Fiscal dominance",
              "monetary_fiscal_coordination" = "Monetary-fiscal coordination",
              "monetary_financial_coordination" = "Moneary-financial coordination")[[varname]])
}

