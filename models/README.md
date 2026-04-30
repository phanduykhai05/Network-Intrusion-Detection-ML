# Models
Thư mục này chứa model đã train.

## File
- `random_forest.pkl` — Random Forest model (best model)

## Lưu ý
- Nếu file `.pkl` > 100MB thì vào gitignore mở ra

- Tải model tại: [Google Drive link] 

## Load model
```python
import joblib
model = joblib.load("models/random_forest.pkl")
```