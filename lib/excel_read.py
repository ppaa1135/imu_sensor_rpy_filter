import openpyxl
import numpy as np

def excel_file_read(file_name):
    wb = openpyxl.load_workbook(filename=file_name)
    ws = wb['Sheet']

    imu_dataset = []

    for idx, row in enumerate(ws.rows):
        if idx == 0:
            continue

        one_imu = []
        for col in row:
            one_imu.append(col.value)

        imu_dataset.append(one_imu)

    imu_dataset = np.array(imu_dataset)
    return imu_dataset
