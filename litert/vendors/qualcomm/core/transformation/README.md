# Graph-to-graph Transformation
## Matmul-convert Fusion
Original graph:
```mermaid
graph TB
  In1 --> MatMul1["MatMul"]
  In2 --> MatMul2["MatMul"]
  MatMul1 --> Convert["Convert"]
  Convert --> |"Quant Params"| Concat["Concat"]
  MatMul2["MatMul"] --> Concat --> Out
  In1@{ shape: sm-circ}
  In2@{ shape: sm-circ}
  Out@{ shape: sm-circ}
```
The updated computational graph after the transformation is shown below.
```mermaid
graph TB
  In1 --> MatMul1["MatMul"]
  In2 --> MatMul2["MatMul"]
  MatMul1 --> |"Quant Params"| Concat["Concat"]
  MatMul2["MatMul"] --> Concat --> Out
  In1@{ shape: sm-circ}
  In2@{ shape: sm-circ}
  Out@{ shape: sm-circ}
```
## Multi-head Attention Optimization
| Dimension	 | Description	   | Example (Gemma3 1B)           |
|:-----------|----------------:|------------------------------:|
| B          | Batch size      | 1                             |
| T          | Sequence length | 128 for prefill/ 1 for decode |
| N          | Number of heads | 4                             |
| K          | Number KV heads | 1                             |
| H          | Head size       | 256                           |
| KV_LEN     | KV Cache Length | 1280                          |
* Notation reference: [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch)

### Gemma3 (Prefill)
Original MHA (Multi-head Attention) in Gemma3 prefill graph
```mermaid
graph TB
  RopeOut --> |"[B, T, N, H]"| Mul["Mul"]
  Mul --> |"[B, T, N, H]"| Transpose["Transpose"]
  Transpose --> |"[B, N, T, H]"| Reshape1["Reshape"]
  Reshape1 --> |"[B, 1, N*T, H]"| MatMul1["MatMul"] & MatMul2["MatMul"]
  MatMul1 --> |"[B, 1, N\*T, K*KV_LEN]"| Concat2
  MatMul2 --> |"[B, 1, N\*T, K*T]"| Concat2
  Concat2[Concat] --> |"[B, 1, N*T, K\*KV_LEN+K\*T]"| Reshape2["Reshape"] 
  Reshape2 --> |"[B, N, T, K\*KV_LEN+K\*T]"| Add["Add"]
  Mask["Mask"] --> |"[B, 1, T, K\*KV_LEN+K\*T]"| Add
  Add --> |"[B, N, T, K\*KV_LEN+K\*T]"| Reshape3["Reshape"] 
  Reshape3--> |"[B, 1, N*T, K\*KV_LEN+K\*T]"| Softmax["Softmax"] 
  Softmax --> |"[B, 1, N*T, K\*KV_LEN+K\*T]"| StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> |"[B, 1, N*T, K\*KV_LEN]"| MatMul3["MatMul"]
  StridedSlice2 --> |"[B, 1, N*T, K\*T]"| MatMul4["MatMul"]
  MatMul3 & MatMul4 --> |"[B, 1, N*T, H]"| Add2["Add"]
  Add2--> |"[B, 1, N*T, H]"| Reshape4["Reshape"]
  Reshape4 --> |"[B, N, T, H]"| Transpose1["Transpose"]
  Transpose1 --> |"[B, T, N, H]"| Reshape5["Reshape"]
  Reshape5 --> |"[B, T, N*H]"| FCIn

  KCache["K Cache"] --> |"[B, 1, K*KV_LEN, H]"| MatMul1
  KSlice["K Slice"] --> |"[B, K*T, 1, H]"| ReshapeKSlice["Reshape"]
  ReshapeKSlice --> |"[B, 1, K*T, H]"| MatMul2
  ReshapeKSlice --> |"[B, 1, K*T, H]"| KSliceOut["K Slice"]

  VCache["V Cache"] --> |"[B, 1, H, K*KV_LEN]"| MatMul3
  VSlice["V Slice"] --> |"[B, K*T, 1, H]"| TransposeVSlice[Transpose]
  TransposeVSlice --> |"[B, 1, H, K*T]"| MatMul4
  TransposeVSlice --> |"[B, 1, H, K*T]"| VSliceOut["V Slice"]

  KCache@{shape: text}
  KSlice@{shape: text}
  KSliceOut@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}
  VSlice@{shape: text}
  VSliceOut@{shape: text}
  FCIn@{shape: sm-circ}
  RopeOut@{shape: sm-circ}
```
The updated computational graph after the transformation is shown below.
```mermaid
graph TB
  RopeOut --> |"[B, T, N, H]"| Transpose["Transpose"]
  Transpose --> |"[B, N, T, H]"| Reshape1
  Reshape1["Reshape"] --> |"[B, 1, N*T, H]"| Split
  Split["Split
  (N=4)"] --> |"[B, 1, T, H]"| SHA1["SHA"] & SHA2["SHA"] & SHA3["SHA"] & SHA4["SHA"]
  SHA1 & SHA2 & SHA3 & SHA4 --> |"[B, 1, T, H]"| Concat2["Concat 
  (N=4)"]
  Concat2 --> |"[B, 1, T, N*H]"| Reshape2[Reshape]
  Reshape2 --> |"[B, T, N*H]"| FCIn
  SHAInputs["SHA Inputs"] --> SHA1 & SHA2 & SHA3 & SHA4
  KSlice["K Slice"] --> ReshapeKSlice[Reshape]
  ReshapeKSlice --> SHAInputs
  ReshapeKSlice --> KSliceOut["K Slice"]
  VSlice["V Slice"] --> TransposeVSlice[Transpose]
  TransposeVSlice --> VSliceOut["V Slice"]
  TransposeVSlice --> SHAInputs

  KCache["K Cache"] --> SHAInputs
  Mask["Mask"] --> SHAInputs
  VCache["V Cache"] --> SHAInputs

  KCache@{shape: text}
  KSlice@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}

  SHAInputs@{shape: text}
  KSliceOut@{shape: text}
  VSliceOut@{shape: text}
  KSlice@{shape: text}
  VSlice@{shape: text}
  FCIn@{shape: sm-circ}
  RopeOut@{shape: sm-circ}
```
with the following four SHA (Single-head Attention), based on the number of heads in MHA.
```mermaid
graph TB
  SHAIn --> |"[B, 1, T, H]"| Mul1
  Mul1["Mul"] --> |"[B, 1, T, H]"| MatMul1["MatMul"] & MatMul2["MatMul"]
  MatMul1 --> |"[B, 1, T, K*KV_LEN]"| Concat["Concat"]
  MatMul2 --> |"[B, 1, T, K*T]"| Concat["Concat"]
  Concat --> |"[B, 1, T, K*KV_LEN+K\*T]"| Add["Add"]
  Add --> |"[B, 1, T, K*KV_LEN+K\*T]"| SoftMax["Softmax"] 
  SoftMax --> |"[B, 1, T, K*KV_LEN+K\*T]"| StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> |"[B, 1, T, K*KV_LEN]"| MatMul3["MatMul"]
  StridedSlice2 --> |"[B, 1, T, K\*T]"| MatMul4["MatMul"]
  KCache["K Cache"] --> |"[B, 1, K*KV_LEN, H]"| MatMul1
  KSlice["K Slice"] --> |"[B, K*T, 1, H]"| ReshapeKSlice[Reshape]
  ReshapeKSlice --> |"[B, 1, K*T, H]"| MatMul2
  ReshapeKSlice --> |"[B, 1, K*T, H]"| KSliceOut["K Slice"]
  Mask["Mask"] --> |"[B, 1, T, K*KV_LEN+K\*T]"| Add
  VCache["V Cache"] --> |"[B, 1, H, K*KV_LEN]"| MatMul3
  VSlice["V Slice"] --> |"[B, K*T, 1, H]"| TransposeVSlice[Transpose]
  TransposeVSlice --> |"[B, 1, H, K*T]"| MatMul4
  TransposeVSlice --> |"[B, 1, H, K*T]"| VSliceOut["V Slice"]
  MatMul3 & MatMul4 --> |"[B, 1, T, H]"| Add2["Add"]
  Add2 --> |"[B, 1, T, H]"| SHAOut
  SHAIn@{shape: sm-circ}
  SHAOut@{shape: sm-circ}
  KCache@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}
  KSlice@{shape: text}
  VSlice@{shape: text}
  KSliceOut@{shape: text}
  VSliceOut@{shape: text}
```

### Multi-head Attention with MaskedSoftmax via Select
The figure below shows multi-head attention with `MaskedSoftmax` implemented using `Select`. All MHAs with `Select` share the same structure and use the same `NotEqual` output as input to their `Select` operations.
```mermaid
graph TB
  Q1 --> |"[B, T, N, H]"| Mul1["Mul"]
  K1 --> |"[B, KV_LEN, N, H]"| Mul2["Mul"]
  Mul1 --> |"[B, T, N, H]"| Transpose1["Transpose"]
  Mul2 --> |"[B, KV_LEN, N, H]"| Transpose2["Transpose"]
  Transpose1 --> |"[B, N, T, H]"| MatMul1["MatMul"]
  Transpose2 --> |"[B, N, H, KV_LEN]"| MatMul1
  MatMul1 --> |"[B, N, T, KV_LEN]"| Select
  Select --> |"[B, N, T, KV_LEN]"| Softmax
  Softmax --> |"[B, N, T, KV_LEN]"| MatMul2["MatMul
  (adj_y = true)"]
  V1 --> |"[B, KV_LEN, N, H]"| Transpose3["Transpose"]
  Transpose3 --> |"[B, N, H, KV_LEN]"| MatMul2
  MatMul2 --> |"[B, N, H, T]"| Transpose4["Transpose"]
  Transpose4 --> |"[B, T, N, H]"| Out
  Mask --> |"[B, T, KV_LEN]"| Reshape
  Reshape --> |"[B, 1, T, KV_LEN]"| NotEqual
  NotEqual --> |"[B, 1, T, KV_LEN]"| Select
  Q2 --> MHA1["MHAs with Select"]
  K2 --> MHA1["MHAs with Select"]
  V2 --> MHA1["MHAs with Select"]
  MHA1 --> Out2
  NotEqual --> MHA1["MHAs with Select"]
  NotEqual --> MHA2["..."]
  Q1@{ shape: text}
  K1@{ shape: text}
  V1@{ shape: text}
  Q2@{ shape: text}
  K2@{ shape: text}
  V2@{ shape: text}
  Mask@{ shape: text}
  Out@{ shape: sm-circ}
  Out2@{ shape: sm-circ}
  MHA2@{ shape: text }
```
The `Reshape → NotEqual → Select` pattern can be optimized through the following operations. It is important to note that the mask produced by the `Mul` operation is reused by all subsequent SHAs involving `Add` for `MaskedSoftmax`.
```mermaid
graph TB
  Mask --> |"[B, T, KV_LEN]"| Equal
  Equal --> |"[B, T, KV_LEN]"| Cast
  Cast --> |"[B, T, KV_LEN]"| Mul
  Mul --> |"[B, T, KV_LEN]"| SHA1["Mask via Add"]
  SHA1@{ shape: text }
  Mask@{ shape: text}
```
With `Mask via Add`, the overall multi-head attention can be transformed to the following SHAs
```mermaid
graph TB
  Q1 --> |"[B, T, N, H]"| Unpack1["Unpack"]
  Unpack1 --> |"[B, T, H]"| MHASHA1["SHA with Add"]
  Unpack1 --> MHASHA2["SHA with Add"]
  Unpack1 --> MHASHA3["..."]
  K1 --> |"[B, KV_LEN, N, H]"| Unpack2["Unpack"]
  Unpack2 --> |"[B, KV_LEN, H]"| MHASHA1["SHA with Add"]
  Unpack2 --> MHASHA2
  Unpack2 --> MHASHA3
  V1 --> |"[B, KV_LEN, N, H]"| Unpack3["Unpack"]
  Unpack3 --> |"[B, KV_LEN, H]"| MHASHA1["SHA with Add"]
  Unpack3 --> MHASHA2
  Unpack3 --> MHASHA3
  Mask --> MHASHA1
  Mask --> MHASHA2
  Mask --> MHASHA3
  Mask --> MHA1["MHAs with Add"]
  Q2 --> MHA1
  K2 --> MHA1
  V2 --> MHA1
  MHA1 --> Out2
  Mask --> MHA2["..."]
  MHASHA1 --> |"[B, T, H]"| Pack
  MHASHA2 --> Pack
  MHASHA3 --> Pack
  Pack --> |"[B, T, N, H]"| Out
  Q1@{ shape: text}
  K1@{ shape: text}
  V1@{ shape: text}
  Q2@{ shape: text}
  K2@{ shape: text}
  V2@{ shape: text}
  Mask(Mask via Add)@{ shape: text}
  Out@{ shape: sm-circ}
  Out2@{ shape: sm-circ}
  MHASHA3@{ shape: text }
  MHA2@{ shape: text }
```
with `SHA with add` below.
```mermaid
graph TB
  Q_unpack --> |"[B, T, H]"| Mul1["Mul"]
  K_unpack --> |"[B, KV_LEN, H]"| Mul2["Mul"]
  Mul2 --> |"[B, KV_LEN, H]"| Transpose2["Transpose"]
  Mul1 --> |"[B, T, H]"| MatMul1["MatMul"]
  Transpose2 --> |"[B, H, KV_LEN]"| MatMul1
  MatMul1 --> |"[B, T, KV_LEN]"| Add
  Mask --> Add
  Add --> |"[B, T, KV_LEN]"| Softmax
  Softmax --> |"[B, T, KV_LEN]"| MatMul2["MatMul"]
  V_unpack --> |"[B, KV_LEN, H]"| MatMul2
  MatMul2 --> |"[B, T, H]"| Out
  Q_unpack@{ shape: text}
  K_unpack@{ shape: text}
  V_unpack@{ shape: text}
  Mask(Mask via Add)@{ shape: text}
  Out@{ shape: sm-circ}
```