import cv2
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


df=pd.read_csv('sampleDataset.csv')


def parse_employment(emp_str):
    try:
        data = ast.literal_eval(emp_str) if isinstance(emp_str, str) else emp_str
        if isinstance(data, list) and len(data) > 0:
            incomes = [item.get('income') for item in data
                       if isinstance(item, dict) and item.get('income') is not None]
            years = [item.get('years') for item in data
                     if isinstance(item, dict) and item.get('years') is not None]
            if incomes and years:
                return pd.Series([float(np.mean(incomes)), float(np.mean(years))])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan])


df[['avg_income', 'avg_years']] = df['Customer Profile.employment_history'].apply(parse_employment)

df['income_level_clean'] = df['Customer Profile.income_level'].fillna(df['avg_income'])
df['dti_ratio'] = df['Loan Application Summary.loan_amount'] / (df['income_level_clean'] * 12 + 1e-5)
df['liquidity_buffer'] = df['Account Activity.balance'] / (df['Loan Application Summary.loan_amount'] + 1e-5)
df['net_cash_flow'] = df['Account Activity.deposits'] - df['Account Activity.withdrawals']


num_cols = ['Account Activity.balance', 'income_level_clean', 'Loan Application Summary.interest_rate',
            'Loan Application Summary.loan_amount', 'dti_ratio', 'liquidity_buffer', 'net_cash_flow']
cat_cols = ['Customer Profile.occupation', 'Loan Application Summary.loan_purpose']

X = df[num_cols + cat_cols].copy()
le = LabelEncoder()
y = le.fit_transform(df['Loan Application Summary.loan_status'])


preprocaster = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocaster),
    ('classifier', GradientBoostingClassifier())
])

model.fit(X, y)

sample_idx = 0
sample_customer = X.iloc[sample_idx:sample_idx + 1]
sample_prediction = model.predict(sample_customer)
approval_idx = list(le.classes_).index('approved')
approval_prob = model.predict_proba(sample_customer)[0][approval_idx]
predicted_status = le.inverse_transform(sample_prediction)[0]
print(f"Predicted status: {predicted_status} | Approval probability: {approval_prob:.2%}")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

frame_count = 0
print("Starting live stream... Press 'q' to quit or 's' to save annotated frame output.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Loan Approval Probability: {approval_prob:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"Customer ID: {df.iloc[sample_idx]['Customer Profile.customer_id'][:8]}...", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"DTI Ratio: {sample_customer['dti_ratio'].values[0]:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Approval Prob: {approval_prob * 100:.1f}%", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if approval_prob > 0.5 else (0, 0, 255), 2)
    cv2.putText(frame, f"Faces Detected: {len(faces)}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Multimodal Fintech eKYC Live Stream', frame)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        output_filename = f"ekyc_output_frame_{frame_count}.png"
        cv2.imwrite(output_filename, frame)
        print(f"Saved annotated frame as {output_filename}")

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_filename = f"face_{frame_count}.png"
            cv2.imwrite(face_filename, face_roi)
            print(f"Detected face saved as {face_filename}")

cap.release()
cv2.destroyAllWindows()

