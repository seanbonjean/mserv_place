import sys

class RedirectStdoutToFileAndConsole:
    """将标准输出同时重定向到文件和控制台"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.file = None

    def start(self):
        """开始重定向"""
        self.file = open(self.file_path, 'w')
        sys.stdout = self  # 替换标准输出为当前类实例

    def stop(self):
        """停止重定向"""
        if self.file:
            self.file.close()
        sys.stdout = self.original_stdout  # 恢复原始标准输出

    def write(self, message):
        """同时写入文件和控制台"""
        if self.file:
            self.file.write(message)
        self.original_stdout.write(message)

    def flush(self):
        """刷新缓冲区"""
        if self.file:
            self.file.flush()
        self.original_stdout.flush()


# 示例使用
if __name__ == "__main__":
    print("这条信息只输出到控制台")  # 控制台输出

    redirector = RedirectStdoutToFileAndConsole("output/test.txt")
    redirector.start()  # 开始重定向

    print("这条信息同时输出到控制台和文件")
    print("再来一条！")

    redirector.stop()  # 停止重定向

    print("结束，输出已恢复到控制台")  # 控制台输出
