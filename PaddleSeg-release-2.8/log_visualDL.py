from visualdl import LogReader

# 你的日志文件路径
log_file_path = r"output/hrsegnetb16" # 请确认这个路径

try:
    reader = LogReader(logdir=log_file_path) # 注意这里可能不是直接文件路径，而是日志文件所在的目录
                                           # 如果vdlrecords.XXX.log在某个目录下，通常指向该目录
    # 遍历所有记录的标量（scalar）数据
    print(f"--- 正在尝试解析 VisualDL 日志: {log_file_path} ---")
    for tag in reader.get_tags().scalar:
        print(f"标签: {tag}")
        for item in reader.get_scalar(tag):
            print(f"  步骤: {item.step}, 值: {item.value}, 墙钟时间: {item.wall_time}")
    print("--- VisualDL 日志解析完成 ---")

except Exception as e:
    print(f"解析 VisualDL 日志时发生错误：{e}")
    print("这可能不是一个 VisualDL 日志文件，或者文件路径不正确。")