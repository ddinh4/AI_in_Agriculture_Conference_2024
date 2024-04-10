df1 = read.csv("G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29/Dina_Dinh/2024_Spring_AIConference/Workshop/Results/model_comparison_results.csv")
t1 = t.test(df1$XGBoost, df1$Random_Forest)
t1
