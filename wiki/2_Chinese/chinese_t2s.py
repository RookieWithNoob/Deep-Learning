"""
    FILE :  chinese_t2s.py
    FUNCTION : None
    繁体中文转换简体中文
    python3 clean_corpus.py --input 1.txt --output 2.txt
"""

import sys
import os
import opencc
from optparse import OptionParser


class T2S(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.cc = opencc.OpenCC('t2s')
        self.t_corpus = []
        self.s_corpus = []
        self.read(self.infile)
        self.t2s()
        self.write(self.s_corpus, self.outfile)

    def read(self, path):
        if os.path.isfile(path) is False:
            print("路径不是文件")
            exit()
        now_line = 0
        with open(path, encoding="UTF-8") as f:
            for line in f:
                now_line += 1
                line = line.replace("\n", "").replace("\t", "")
                self.t_corpus.append(line)
        print("读取完毕")

    def t2s(self):
        now_line = 0
        all_line = len(self.t_corpus)
        for line in self.t_corpus:
            now_line += 1
            if now_line % 1000 == 0:
                sys.stdout.write("\r已经处理{} 行, 一共有 {} 行.".format(now_line, all_line))
            self.s_corpus.append(self.cc.convert(line))
        sys.stdout.write("\r已经处理{} 行, 一共有 {} 行.".format(now_line, all_line))
        print("\n处理完毕")

    def write(self, list, path):
        print("正在写入新文件......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for line in list:
            file.writelines(line + "\n")
        file.close()
        print("写入完成.")


if __name__ == "__main__":
    print("繁体中文到简体中文")
    # input = "./wiki_zh_10.txt"
    # output = "wiki_zh_10_sim.txt"
    # T2S(infile=input, outfile=output)

    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="traditional file")
    parser.add_option("--output", dest="output", default="", help="simplified file")
    (options, args) = parser.parse_args()

    input = options.input
    output = options.output

    try:
        T2S(infile=input, outfile=output)
        print("全部完成.")
    except Exception as err:
        print(err)
