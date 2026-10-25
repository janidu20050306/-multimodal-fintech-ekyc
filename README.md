# Fintech eKYC Risk Dashboard

A real-time multimodal demo that combines a scikit-learn loan-approval risk model with OpenCV webcam face detection. The live camera feed is annotated with detected faces, the sample customer's debt-to-income ratio, and the model's approval probability.

> Educational demo only. This is not a production credit-scoring engine, a real eKYC/identity-verification system, or financial advice.

## Features

- End-to-end ML pipeline built with scikit-learn `Pipeline` and `ColumnTransformer`
- Feature engineering for credit risk: debt-to-income ratio, liquidity buffer, and net cash flow
- Employment-history parsing into average income and tenure
- Live webcam overlay with Haar-cascade face detection
- Save annotated frames and cropped face images with a keypress
- Graceful handling of unseen categorical values via `OneHotEncoder(handle_unknown="ignore")`

## How It Works

1. Load `sampleDataset.csv`.
2. Parse the `Customer Profile.employment_history` column and derive average income and years of employment.
3. Engineer risk features:
   - `dti_ratio = loan_amount / (monthly_income * 12)`
   - `liquidity_buffer = balance / loan_amount`
   - `net_cash_flow = deposits - withdrawals`
4. Preprocess numeric columns with mean imputation and categorical columns with one-hot encoding.
5. Train a `GradientBoostingClassifier` to predict `Loan Application Summary.loan_status` (`approved`, `pending`, `rejected`).
6. Score one sample customer and overlay the approval probability on the webcam feed.

### Model Inputs

| Type | Features |
| --- | --- |
| Numeric | balance, income level, interest rate, loan amount, DTI ratio, liquidity buffer, net cash flow |
| Categorical | occupation, loan purpose |

The approval probability is the predicted probability of the `approved` class, looked up by class name rather than a hard-coded index.

## Dataset

`sampleDataset.csv` contains 1,000 rows and 34 columns covering customer profiles, account activity, loan applications, and time-series data.

The target column is `Loan Application Summary.loan_status`:

| Class | Rows |
| --- | --- |
| approved | 448 |
| pending | 284 |
| rejected | 268 |

The CSV is about 22 MB. Before publishing, confirm that the data is synthetic or properly anonymized, and consider Git LFS or a smaller sample if needed.

## Requirements

- Python 3.10+ (tested on 3.14)
- A webcam
- `opencv-python` 4.x, `pandas`, `numpy`, `scikit-learn`

OpenCV 5.x removed `cv2.CascadeClassifier` and the bundled Haar cascades, so this project pins OpenCV below 5.

## Installation

```bash
git clone https://github.com/<your-username>/fintech-ekyc-risk-dashboard.git
cd fintech-ekyc-risk-dashboard

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install "opencv-python>=4.8,<5" pandas numpy scikit-learn
```

## Usage

Place `sampleDataset.csv` in the project root, then run:

```bash
python main.py
```

The script trains the model, prints the predicted status and approval probability for the sample customer, and opens the live stream window.

| Key | Action |
| --- | --- |
| `s` | Save the annotated frame and any detected face crops |
| `q` | Quit |

Saved files:

- `ekyc_output_frame_<n>.png`
- `face_<n>.png`

To score a different customer, change `sample_idx` in `main.py`. To use a different camera, change `cv2.VideoCapture(0)`.

## Project Structure

```text
fintech-ekyc-risk-dashboard/
|-- main.py
|-- sampleDataset.csv
`-- README.md
```

## Limitations

- Haar cascades detect frontal faces only and are sensitive to lighting, pose, and occlusion. There is no face recognition, liveness detection, or identity matching.
- The model is trained and scored on the same data, with no train/test split, calibration, or fairness evaluation. The displayed probability is illustrative.
- `pending` is treated as a separate class, so the demo does not model a binary approve/reject decision boundary.
- The sample dataset includes customer and account fields that may be sensitive. Review and anonymize before making the repository public.

## Roadmap

- Add a train/test split with ROC-AUC, precision/recall, and calibration metrics
- Replace Haar cascades with a modern detector such as YuNet or MediaPipe
- Add liveness checks for a more realistic eKYC flow
- Move configuration into CLI arguments or a config file
- Add tests for feature engineering and inference

## License

MIT is a good default for a public demo repository. Add a `LICENSE` file before publishing.
