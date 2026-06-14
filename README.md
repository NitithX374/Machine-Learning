# 🧠 Machine Learning จากศูนย์ — ชุด Notebook เรียนรู้ด้วยตัวเอง

ชุด notebook สอน Machine Learning โดย **เขียนเองทุกบรรทัด** ตั้งแต่ neuron เดียว ไปจนถึง LLM, การตรวจจับภัยเครือข่าย, agent ที่เรียนรู้เอง และการสร้างภาพ

> **แนวคิดเดียวที่ร้อยทุกเล่มเข้าด้วยกัน:** โมเดลทุกตัว ตั้งแต่เส้นตรงเส้นเดียวจนถึง GPT ทำงานด้วยวงจรเดียวกัน —
> **forward (คำนวณคำตอบ) → loss (วัดความผิด) → backprop (หา gradient) → update (ปรับ weight) → วนซ้ำ**
> สิ่งที่ต่างกันมีแค่ "สถาปัตยกรรม" ที่เอา neuron มาประกอบ และ "loss วัดอะไร" เท่านั้น

ทุกเล่มคอมเมนต์ภาษาไทย, รันได้จริงบน CPU (มี output/กราฟฝังในไฟล์), ไม่ใช้ library สำเร็จรูปสำหรับแก่นของแต่ละเรื่อง

---

## 📚 ลำดับการอ่านที่แนะนำ

อ่านเรียงตามนี้จะต่อเนื่องที่สุด (แต่ละเล่มอ้างอิงแนวคิดจากเล่มก่อน):

```
1. Fundamentals → 2. LLM Deep Dive → 3. Network Security → 4. Reinforcement Learning → 5. Generative Vision → 6. Graph Neural Nets
   (พื้นฐาน)        (ต่อยอด LLM)       (เอาไปใช้จริง)         (paradigm ที่ 3)            (สร้างภาพ)            (ข้อมูลกราฟ)

7. ML Pitfalls (ทำ ML ให้ถูก) — อ่านได้ทุกเมื่อหลังเล่ม 1/3 เป็นเล่ม "วิธีใช้ให้ไม่พลาด"
```

ถ้าสนใจเรื่องไหนเป็นพิเศษก็กระโดดอ่านได้ แต่ละเล่มมีสรุปแนวคิดที่ใช้ในตัว

> 📖 เจอศัพท์ที่ไม่คุ้น? เปิด [**GLOSSARY.md**](GLOSSARY.md) — รวมศัพท์ ML ทั้งหมดที่ใช้ในชุดนี้ พร้อมความหมายไทยและบอกว่าอยู่เล่มไหน

---

## 1. 🧠 `ML_Fundamentals.ipynb` — พื้นฐานทั้งหมด

**เริ่มจากอะไร:** neuron ตัวเดียว → **ถึงไหน:** mini GPT ที่ generate ข้อความได้

| Part | เนื้อหา |
|---|---|
| 0 | Linear regression — gradient descent ที่เห็นภาพง่ายสุด |
| 1 | Neuron เดี่ยว และทำไมมันแก้ XOR ไม่ได้ |
| 2 | MLP แก้ XOR — เขียน backprop เองด้วย **pure Python** (ไม่มีแม้แต่ NumPy) |
| 3 | เวอร์ชัน NumPy — คิดเป็น matrix (vectorization) |
| 4 | TensorFlow/Keras + MNIST (จำแนกเลขลายมือจริง) |
| 5 | Overfitting / Underfitting — ปัญหากลางของ ML ทั้งวงการ |
| 6 | CNN — สถาปัตยกรรมสำหรับรูปภาพ + confusion matrix |
| 7 | Attention mechanism — Q/K/V, causal mask, multi-head (เขียนเอง) |
| 8 | Mini GPT — สร้าง transformer จิ๋วระดับตัวอักษร แล้ว generate จริง |

**จะได้รู้:** backprop / chain rule ทำงานยังไงจริง ๆ, ทำไมต้องมี hidden layer, ทำไม deep learning = การคูณ matrix (เลยต้องใช้ GPU), attention คืออะไร, LLM ทำงานยังไงในระดับพื้นฐาน

---

## 2. 🔬 `LLM_Deep_Dive.ipynb` — แกะ LLM ทีละชิ้น

**ต่อจากเล่ม 1 (mini GPT)** เจาะลึกชิ้นส่วนรอบ ๆ ที่ทำให้ LLM จริงทำงานได้

| Part | เนื้อหา |
|---|---|
| 1 | **BPE Tokenizer** — เขียน Byte Pair Encoding จากศูนย์ (ทำไม LLM นับตัวอักษรใน "strawberry" พลาด) |
| 2 | Pretrain mini GPT บน BPE token + ดู embedding ที่เรียนได้ |
| 3 | **Sampling** — greedy / temperature / top-k / top-p ต่างกันยังไง |
| 4 | **KV Cache** — ทำไม LLM generate ได้เร็ว (วัดเวลาจริง เร็วขึ้น ~12 เท่า) |
| 5 | **Fine-tuning** — เปลี่ยน base model เป็นผู้ช่วยตอบคำถาม + catastrophic forgetting |
| 6 | จากของเล่นสู่ Claude/GPT จริง — RLHF, RAG, long context, agents |

**จะได้รู้:** tokenizer ทำงานยังไง, พารามิเตอร์ `temperature`/`top_p` ที่เจอตอนเรียก API คืออะไร, fine-tuning vs pretraining, ทำไมโมเดลถึง "ลืม" ของเดิมถ้าเทรนแรงเกิน

---

## 3. 🛡️ `ML_Network_Security.ipynb` — ML กับความปลอดภัยเครือข่าย

เอา ML ไปใช้งานจริง + เรียน **classical ML** และ **unsupervised learning** ที่ยังไม่ได้แตะในเล่มก่อน

| Part | เนื้อหา |
|---|---|
| 1 | Network flow ในสายตา ML (feature engineering) |
| 2 | จำแนกประเภท traffic — **Decision Tree → Random Forest** (วาดต้นไม้ให้ดู) |
| 3 | ตรวจจับการบุกรุก (IDS) — **Isolation Forest + Autoencoder** (เรียนจาก traffic ปกติเท่านั้น) |
| 4 | ตรวจจับโดเมน DGA ของ botnet (char-CNN) + การหลบหลีกโมเดล |
| 5 | **Adversarial ML** — FGSM โจมตีโมเดลของเราเอง (noise จิ๋วที่ตามองไม่เห็น) |
| 6 | จากแล็บสู่ของจริง — dataset จริง (CICIDS2017), Zeek, ความท้าทาย |

**จะได้รู้:** ต้นไม้ตัดสินใจ/random forest, ทำไมงานข้อมูลตารางไม่ต้องใช้ neural network, anomaly detection แบบไม่มี label, ทำไม accuracy หลอกตาได้, โมเดลถูกหลอกด้วย adversarial example ยังไง

---

## 4. 🎮 `ML_Reinforcement_Learning.ipynb` — เรียนรู้จากการลองผิดลองถูก

เสาหลักที่ 3 ของ ML (ถัดจาก supervised/unsupervised) — agent เรียนเองจาก **reward**

| Part | เนื้อหา |
|---|---|
| 1 | RL คืออะไร + สร้าง **Gridworld** เอง (agent เดินเก็บ reward) |
| 2 | **Q-Learning** (ตาราง) — Bellman equation, explore vs exploit |
| 3 | **DQN** — พอ state เยอะเกินใส่ตาราง ใช้ neural net แทน (เล่น **CartPole** ที่เขียน physics เอง) |
| 4 | **REINFORCE** (policy gradient) — เรียน policy ตรง ๆ |
| 5 | เชื่อมกับ **RLHF** — RL ฝึก LLM ยังไง (ปิดวงจากเล่ม 2) |
| 6 | ของจริง (AlphaGo, หุ่นยนต์) + สรุป |

**จะได้รู้:** agent/state/action/reward, Bellman equation, ความต่างของ value-based (DQN) vs policy-based (REINFORCE), RLHF ที่ใช้ฝึก ChatGPT/Claude ทำงานบนหลักการเดียวกับที่เขียนเองนี่แหละ

---

## 5. 🎨 `ML_Generative_Vision.ipynb` — สอนเครื่องให้ "วาด" ภาพ

generative model ฝั่ง*ภาพ* (เล่ม 2 ทำฝั่งข้อความไปแล้ว) — สร้าง 3 ตระกูลหลักบน MNIST

| Part | เนื้อหา |
|---|---|
| 2 | **VAE** — บีบเป็น latent แล้วสุ่มคืน (เห็น latent manifold สวย ๆ แต่ภาพเบลอ) |
| 3 | **GAN** — generator ปะทะ discriminator (ภาพคมขึ้น แต่เทรนยาก) |
| 4 | **Diffusion** — เติม noise แล้วเรียนถอย (เบื้องหลัง Stable Diffusion / DALL·E) |
| 5 | เทียบ 3 ตระกูล + เชื่อม autoregressive (เล่ม 2) |

**จะได้รู้:** ความต่างของ generative vs discriminative, reparameterization trick, การแข่งกันของ GAN, กลไก diffusion (forward เติม noise / reverse ถอย noise), ทำไม Stable Diffusion ถึงสร้างภาพได้

> 📝 **หมายเหตุตามจริง:** โมเดลในเล่มนี้จิ๋วและรันบน CPU ภาพที่ได้จึงหยาบ — โดยเฉพาะ diffusion ที่ generate
> จาก noise ล้วนได้แค่เค้าโครง (โมเดลเล็ก error สะสมตลอด 200 ขั้น) เล่มจึงโชว์ผ่าน "การ denoise ภาพจริง"
> ที่ออกมาคมชัด เพื่อให้เห็นกลไกอย่างซื่อสัตย์ ของจริงต้องใช้ GPU + โมเดลใหญ่ + ข้อมูลมหาศาล

---

## 6. 🕸️ `ML_Graph_Neural_Networks.ipynb` — เรียนรู้จาก "ความเชื่อมโยง"

สถาปัตยกรรมสำหรับข้อมูลที่เป็น **กราฟ** (node เชื่อมกันด้วย edge) — เครือข่าย, โซเชียล, โมเลกุล

| Part | เนื้อหา |
|---|---|
| 1 | Graph คืออะไร + adjacency matrix + ทำไม ML ปกติไม่พอ |
| 2 | **Message Passing** — หัวใจ GNN เขียนเองด้วย NumPy (สัญญาณแพร่ทีละ hop) |
| 3 | **GCN** จำแนก node กึ่ง supervised — Karate Club (รู้ฝ่ายแค่ 2 คน → ทำนายถูก 97%) |
| 4 | **ตรวจจับ botnet จากโครงสร้างกราฟล้วน ๆ** (feature เหมือนกันหมด — จับได้ 100%) |
| 5 | Over-smoothing (ทำไมซ้อนชั้นลึกไม่ได้) + 3 ระดับงาน GNN |
| 6 | ของจริง (Google Maps, ค้นพบยา, ตรวจฉ้อโกง) + สรุป |

**จะได้รู้:** กราฟต่างจากข้อมูลตาราง/ภาพยังไง, message passing (ให้ node ดูดข้อมูลจากเพื่อนบ้าน), ทำไม label น้อยนิดก็พอเพราะ "ความเชื่อมโยง" เติมเต็มข้อมูล, การมองโครงสร้างทั้งภาพเห็นสิ่งที่ดูทีละจุดไม่เห็น

---

## 7. ⚠️ `ML_Pitfalls.ipynb` — ทำ Machine Learning ให้ "ถูก"

ไม่ใช่อัลกอริทึมใหม่ แต่เป็นเล่ม **"วิธีใช้ให้ไม่พลาด"** — ทักษะที่แยกคน "รู้อัลกอริทึม" กับคน "ทำ ML เป็น"

| Part | กับดัก |
|---|---|
| 1 | **Data Leakage** — ได้ 87% บน noise ล้วน! (เลือก feature ก่อน split) + oversample ผิดที่ |
| 2 | **Cross-Validation** — split เดียวผันผวน 12% ต้องใช้ k-fold รายงานเป็นค่าเฉลี่ย ± SD |
| 3 | **เลือก metric ผิด** — accuracy เท่ากัน 95% แต่ AUC เผย baseline 0.5 vs RF 0.97 |
| 4 | **Calibration & Threshold** — "ความมั่นใจ" ที่เชื่อได้ไหม + default 0.5 ไม่ใช่คำตอบเสมอ |
| 5 | **Overfitting to test set** — เลือกตัวดีสุดจาก 150 โมเดลบน noise → 59% (ลวง) → 49% (จริง) |
| 6 | **Checklist** เช็คก่อนเชื่อผลลัพธ์ + สรุปทั้ง 7 เล่ม |

**จะได้รู้:** ทำไมผลที่ "ดีเกินจริง" มักมี leakage ซ่อนอยู่, การประเมินผลให้ตรงความจริง, วินัยเรื่อง test set, การไม่หลอกตัวเอง — สิ่งที่ทำให้ตัวเลขในงานจริงเชื่อถือได้

---

## 🧪 Interactive Visualizers (เว็บ — เล่นได้ในเบราว์เซอร์)

โฟลเดอร์ [`docs/`](docs/) มีเครื่องมือ interactive ที่ "เล่นแล้วเข้าใจ" — HTML ไฟล์เดียวต่อตัว ไม่ต้องลงอะไร:

| Visualizer | เล่นอะไร |
|---|---|
| **MLP** | ดู neural net เรียนแก้ XOR — decision boundary ก่อตัวสด ปรับ hidden / learning rate ได้ |
| **Backpropagation** | forward คำนวณคำตอบ → gradient ไหลย้อนกลับทีละชั้นด้วย chain rule (ตัวเลขจริง) |
| **Decision Tree** | ต้นไม้แตกกิ่งแบ่งข้อมูลทีละคำถาม จนแต่ละกลุ่มบริสุทธิ์ |
| **XGBoost** | gradient boosting เพิ่มต้นไม้ทีละต้นมาแก้ residual — error ลดลงเรื่อย ๆ |
| **CNN** | convolution → pooling → flatten → dense ทีละขั้น |
| **Transformer** | self-attention Q/K/V ทีละขั้น |

**เปิดดู:** ดับเบิลคลิก `docs/index.html` หรือเอาขึ้น **GitHub Pages** —
ที่ repo: Settings → Pages → Source: *Deploy from a branch* → Branch `main`, Folder `/docs` → Save
แล้วเข้าได้ที่ `https://<username>.github.io/<repo>/`

---

## 🗺️ ภาพรวม: ครอบคลุมอะไรบ้าง

| | Notebook |
|---|---|
| **Supervised learning** | 1, 3, 6 |
| **Unsupervised learning** | 3 (anomaly detection), 5 (generative) |
| **Reinforcement learning** | 4 |
| **Deep learning** (CNN, attention) | 1 |
| **Generative — ข้อความ** | 2 |
| **Generative — ภาพ** | 5 |
| **Security / robustness** | 3, 6 |
| **โครงสร้างข้อมูลที่รองรับ** | เวกเตอร์/ภาพ/ลำดับ (1,2) · ตาราง (3) · ภาพ (5) · **กราฟ (6)** |

ครบทั้ง **3 paradigm หลักของ ML** + generative ทั้งสองฝั่ง + ครบทุกโครงสร้างข้อมูลหลัก + การนำไปใช้จริง

---

## ⚙️ การรัน

ต้องมี Python 3.12 + ไลบรารีเหล่านี้:

```
numpy  matplotlib  tensorflow  scikit-learn  pandas  networkx  jupyter  nbformat
```

เปิดด้วย Jupyter / VS Code แล้วกด **Run All** ได้เลย (output ฝังในไฟล์อยู่แล้ว ไม่ต้องรันก็อ่านได้)
เวลารันต่อเล่ม: ส่วนใหญ่ ~3–6 นาทีบน CPU, เล่ม 6 (GNN) ~1–2 นาที, เล่ม 5 (generative) ~10–12 นาที เพราะเทรน 3 โมเดล

> ฟอนต์: notebook ตั้ง `Tahoma` ไว้สำหรับ label ภาษาไทยในกราฟ (มากับ Windows) ถ้าใช้ OS อื่นเปลี่ยนเป็นฟอนต์ไทยที่มีในเครื่อง

---

*ไฟล์ `XOR_MLP.ipynb`, `xor_mlp.py`, `xor_mlp_numpy.py` เป็นไฟล์ทดลองตั้งต้น (จุดเริ่มของเล่ม 1)*
