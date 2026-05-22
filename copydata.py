import pandas as pd

# Bước 1: Đọc file gốc
df = pd.read_csv('netflix_user_behavior_dataset.csv')

# Bước 2: Lưu ngay ra một file mới y hệt
df.to_csv('netflix_backup.csv', index=False) 

print("Đã tạo bản sao y hệt với tên: netflix_backup.csv")
