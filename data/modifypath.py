import pandas as pd

# โหลดไฟล์ CSV
train_df = pd.read_csv('train.csv')
validate_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# แก้ไข path ใน column 'skincap_file_path'
def correct_path(file_path, folder):
    # แทนที่ path ของไฟล์จาก D:\multimodalll\CNN+NLP\data\test\ เป็น D:\CNN\data\test\
    new_path = file_path.replace("D:\\multimodalll\\CNN+NLP\\data", "D:\CNN\data")
    return new_path

# แก้ไขไฟล์ต่างๆ โดยการแทนที่ path
train_df['skincap_file_path'] = train_df['skincap_file_path'].apply(lambda x: correct_path(x, 'train'))
validate_df['skincap_file_path'] = validate_df['skincap_file_path'].apply(lambda x: correct_path(x, 'validate'))
test_df['skincap_file_path'] = test_df['skincap_file_path'].apply(lambda x: correct_path(x, 'test'))

# บันทึกไฟล์ CSV ที่แก้ไขแล้ว
train_df.to_csv('train_modified.csv', index=False)
validate_df.to_csv('validate_modified.csv', index=False)
test_df.to_csv('test_modified.csv', index=False)

"Files have been updated and saved successfully!"
