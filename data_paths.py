# data_paths.py
import platform

os_name = platform.system()
if os_name == "Darwin":
    BASE_PATH = '/Users/hyunseoki/workspace/src/leak_detection/one_way_protonet/data/cyclegan/data_a'
elif os_name == "Linux":
    BASE_PATH = '/home/hyunseoki_rtx3090/ssd1/02_src/leak_detection/protonet_anomaly/data/cyclegan/data_a'

NORMAL_DATA_PATHS = {
    'a': f'{BASE_PATH}/data_aa/normal/A_20221030_normal.dat',
    'b': f'{BASE_PATH}/data_bb/normal/B_20230407092944_normal.dat',
    'c': f'{BASE_PATH}/data_cc/normal/C_0502_normal_infobee.csv',
}

LEAK_DATA_PATHS = {
    'a': f'{BASE_PATH}/data_aa/leak/A_20221030_leak.dat',
    'b': f'{BASE_PATH}/data_bb/leak/B_20230407095433_leak.dat',
    'c': f'{BASE_PATH}/data_cc/leak/C_0502_leak_infobee.csv',
}
