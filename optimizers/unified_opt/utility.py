# -*- coding: utf-8 -*-
# @Project : lab
# @Time    : 2025/4/2 19:07
# @Author  : jo-xin
# @File    : utility.py
import os
from datetime import datetime


def format_time(seconds):
    seconds = int(seconds)
    # 计算天数
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)

    # 计算小时数
    hours = seconds // 3600
    seconds %= 3600

    # 计算分钟数
    minutes = seconds // 60
    seconds %= 60

    # 构造输出字符串
    time_str = ""
    if days > 0:
        time_str += f"{days} 天"
    if hours > 0:
        if time_str:
            time_str += " "
        time_str += f"{hours} 小时"
    if minutes > 0:
        if time_str:
            time_str += " "
        time_str += f"{minutes} 分"
    if seconds > 0 or not time_str:  # 如果还有剩余秒数或者没有其他单位，则添加秒数
        if time_str:
            time_str += " "
        time_str += f"{seconds} 秒"

    return time_str

log_dir = 'txt_log'

if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)

_log_file = os.path.join(log_dir, 'log' + datetime.now().strftime("%Y_%m_%d___%H_%M_%S") + '.txt')

def do_log(*arg, log_file=_log_file, is_title=False):
    content = ' '.join(str(i) for i in arg)
    if is_title:
        content = '=' * 40 + '\n' + content + '\n' + '=' * 40
    print(content)
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(content + '\n')


