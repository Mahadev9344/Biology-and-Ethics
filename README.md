Project Title : Off-Target Prediction in Cancer Therapy Using CRISPR-Cas9 
CRISPR-Cas9 is a gene-editing tool used to correct genetic mutations in cancer therapy. However, while targeting the correct gene, CRISPR sometimes makes unintended changes (called off-target effects) in the genome, which can cause harmful side effects like cancer cell mutation or other diseases.
To predict off-target effects of CRISPR-Cas9 , We used Feedforward Neural Networks (FNN)

Dataset Used:
-> encodedwithoutTsai dataset (Off-target sequences)
-> guideseq dataset (Validated off-targets from biological experiments)

Preprocessing Techniques:
-> 8-bit Encoding
-> Reshaping into 8x23 format

Model Used:
-> Feedforward Neural Network (FNN) 

Progress So far : 
-> Downloaded Dataset.
-> Done Preprocessing which involves cleaning the dataset and 8-bit One-hot encoding.
-> Build FNN model.
-> Acheived 96% accuracy

