# Recursive-Reasoning

In this project, I have reimplemented 

For generating trees run the following code : 

```
 > python3 main_gen2.py --dataset_name com2sense --mode normal --seed 21
```

For inference on the generated trees : 

```
  > python3 main_inference2.py --dataset_name com2sense --mode normal --seed 21
```

## Results

Dataset : Com2sense (dev set - 782 examples)
| Generator+Verifier |  Generation-Seed  | Inference-Seed  | Type-of-method |    Max-Acc |
| ------------------ | ------------------ | ----------------- | ---------------- | -------------- |
| FlanT5 + Roberta. |   42  |              Default    |             Method-1   |            61.5% |
| FlanT5 + Roberta. |   42     |           Default       |          Method-2    |           66.8% |
| FlanT5 + FlanT5.   |   42     |         42       |                  Method-1     |           58.8% |
| FlanT5 + FlanT5.  |    42     |           15      |                    Method-1   |             60.35% |
| FlanT5 + FlanT5.  |    42     |            42     |                    Method-2   |             69.43% |
| FlanT5 + FlanT5.  |    42      |          21        |                  Method-2      |          70.33% |
| GPT-3 + Roberta (paper)  | | | |                                                                                   72.5%|




Dataset : CREAK (dev set - 1371 examples)
| Generator+Verifier  |  Generation Seed  | Inference Seed  | Type of method  |   Max-Acc |
| ------------------- | ------------------ | ----------------- | ---------------- | ---------- |
| FlanT5 + Roberta. |    42    |                       Default        |         Method-1     |          70.02%  |
|FlanT5 + Roberta.  |  42        |                   Default       |          Method-2       |        82.4% |
|FlanT5 + FlanT5.   |   42          |                 42             |            Method-1      |          70.45% |
|FlanT5 + FlanT5.  |    42           |                42            |             Method-2     |           83.37% |
| FlanT5 + FlanT5. |      42         |                  21          |                Method-1   |             70.45% |
| FlanT5 + FlanT5. |     42          |                 21           |              Method-2     |           83.95% |
| GPT-3 + Roberta (paper) | | | |                                                                                    85.2% |

Dataset - StrategyQA (dev set - 229 examples)
| Generator+Verifier | Seed Value | Type of Method | Max Acc |
| ------------------ | ----------- | ---------------- | --------- |
| Flan T5 + Roberta Verifier | 42 | Method-1 | 50.21 % | 
| Flan T5 + Roberta Verifier | 21 | Method-1 | 50.21 % |
| **Flan T5 + Roberta Verifier** | 42 | Method - 2 | **51.96 %** |
| Flan T5 + Roberta Verifier | 21 | Method - 2 | 51.09 % |
| Flan T5 + Flan T5 Verifier | 42 | Method - 1 | 49.34 % |
| Flan T5 + Flan T5 Verifier | 21 | Method - 1 | 48.03 % |
| Flan T5 + Flan T5 Verifier | 42 | Method - 2 | 51.52 % |
| Flan T5 + Flan T5 Verifier | 21 | Method - 2 | 51.52 % |
| GPT-3 (paper) | | | 60.7% |

Paper Citation: 

```
@inproceedings{Jung2022MaieuticPL,
  title={Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations},
  author={Jaehun Jung and Lianhui Qin and Sean Welleck and Faeze Brahman and Chandra Bhagavatula and Ronan Le Bras and Yejin Choi},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```

