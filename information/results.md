# Training Analysis and Voice recongnition model results

In this document we share our detailed analysis on the performance of our model **Catalan Speech2text**. We have decided to divide the anlysis on three different parts in order to make easier the understanding of each key part of the process:

1. **Loss performance during the training**, based on the plot `train_loss`.
2. **Quantitative and qualitative evaluations on our predictions**, based on specific transcriptions.
3. **Possible explanations for the performance**, considering factors like data, architecture and configurations for the training.

With this model we can clearly see a huge upgrade from the previous implementation

`Before`
```
--------------------------------------------------------
Predicted: oi a
Target: Coneixes algu que visqui a gatova
--------------------------------------------------------
```
`After`
```
--------------------------------------------------------
Predicted: coneixes elguc avisia galtobo
Target: Coneixes algu que visqui a gatova
--------------------------------------------------------
```

## Loss plot (`train_loss`)

The plot of the Traininng Loss shows the expected behaviour. The main characteristics are the following ones:

1. **Steep initial drop**: la `loss` the loss starts above 30 and quickly falls to 3 in the first few iterations, which is common in the early stages of training.
2. **Smooth convergence**: The loss progressively gets lower in each epoch until it stabilizes.

- - -

##  Inference results

The metrics used for the error calculation are WER (Word Error Rate) and CER (Character Error Rate).

After 10 epochs of training we obtained:

| Mean Train Loss | Mean Validation Loss | Mean WER | Mean CER | 
| --------------- | -------------------- | -------- | -------- | 
| 0.61      |0.51           |0.48|0.16|

![W B Chart 23_5_2025, 8_00_03](https://github.com/user-attachments/assets/4508dce2-af43-431b-a580-c6d10a43e397)
![W B Chart 23_5_2025, 7_59_26](https://github.com/user-attachments/assets/a0f0a1a9-54d9-4893-ba18-920147bda9ad)
![W B Chart 23_5_2025, 7_59_46](https://github.com/user-attachments/assets/0d6e20cf-fcc1-4c98-9226-9a1c4f43c59f)
![W B Chart 23_5_2025, 7_59_55](https://github.com/user-attachments/assets/0ed9f5bb-f388-43b3-a6ac-8213bc6cc2df)


In order to analyse the results on a deeper level we have done an evaluation of the output based on the accent depending on where the speaker comes from, and the results are the following ones:

| Accent | WER (%) | CER (%) |
| ------ | ------- | ------- |
| Central | 0.4078 | 0.1301 |
| Balearic | 0.7238 | 0.3004 |
| Valencian | 0.5628 | 0.1783 |
| Northwestern | 0.4649 | 0.1432 |
| Tortosí | 0.4245 | 0.1363 |
| Alguerese | 0.4605 | 0.1501 |
| Ribagorçan | 0.4439 | 0.2086 |
| Rossellonès | 0.5114 | 0.2094 |

- - -

As it can be seen, the central accent has the greatest performance while the Balearic one performs the worst.

##  Qualitative analysis

After doing some observations on the predictions done with the model we can highlight some commons mistakes in the transcriptions:

#### Problems identifying "B" and "V"
```
--------------------------------------------------------
Predicted: els vacteis qui creixenn colitius liquiptsovipormans es pasi es colairals
Target: els bacteris que creixen en cultius liquids sovint formen suspensions colloidals
--------------------------------------------------------
```

#### Does not detect the phoneme "LL"
```
--------------------------------------------------------
Predicted: velemun ana i emn botes que caminam portan una motxila suto m paraivondie de plouja
Target: veiem una noia amb botes que camina portant una motxilla sota un 
--------------------------------------------------------
```
#### Problems identifying neutral vowels 

```
--------------------------------------------------------
Predicted: les teva capital i municipi principal es sart
Target: La teva capital i municipi principal es sort
--------------------------------------------------------
```

#### Concatenation when a word ends like the following one starts
```
--------------------------------------------------------
Predicted: toteixoportestiman qequestemencrementeraels sinmes sonors de lesona
Target: tot aixo porta a estimar que aquesta no incrementara els nivells sonors de la zona
--------------------------------------------------------
```

#### Doesn't detect the silent letters (such as "T" after an "N" at the end of the word, or a silent "H")

```
--------------------------------------------------------
Predicted: i a
Target: hi ha
--------------------------------------------------------
```

#### The phoneme "SS" appears commonly substituted by "C", "Z" or "S"
```
--------------------------------------------------------
Predicted: per totsans mortons ieglans cames seques i is pepessans
Target: per tots sants  murtons i aglans  cama seques i esclata sangs
--------------------------------------------------------
```

# Conclusions

 * Although the model performances correctly and transcribes reasonably well, still has common mistakes, probably caused by a lack of a better tokenization that could take into account vowels with accents, punctuation marks and other signals that could bring more context.

 * The model transcribes correctly audios with neutral (central) accent but fails to do so with extracomunitarian accents, as the valencian and the balearic.

 * Even though this failed trascriptions can be found, we have succeeded on creating a model that is able to perform a good voice recognition with quite well precision results.
