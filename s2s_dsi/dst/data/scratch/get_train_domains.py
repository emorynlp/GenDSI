
import dst.data.dst_data as dst


train_data = dst.DstData.load('data/gptdst5k/gptdst5k_test_domains_0.pkl')

for dialogue in train_data.dialogues:
    ...