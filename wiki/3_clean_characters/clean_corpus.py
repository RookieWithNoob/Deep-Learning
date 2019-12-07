"""
    FILE :  clean_corpus.py
    FUNCTION : None
    包含很多的英文，日文，德语，中文标点，乱码等一些字符，我们要把这些字符清洗掉，只留下中文字符characters
    python3 clean_corpus.py --input zhwiki.txt --output zhwiki_cleaned.txt
"""
import sys
import os
from optparse import OptionParser


class Clean(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.corpus = []
        self.remove_corpus = []
        self.read(self.infile)
        self.remove(self.corpus)
        self.write(self.remove_corpus, self.outfile)

    def read(self, path):
        print("正在读取......")
        if os.path.isfile(path) is False:
            print("path is not a file")
            exit()
        now_line = 0
        with open(path, encoding="UTF-8") as f:
            for line in f:
                now_line += 1
                line = line.replace("\n", "").replace("\t", "")
                self.corpus.append(line)
        print("读取完成.")

    def remove(self, list):
        print("正在清理......")
        for line in list:
            re_list = []
            for word in line:
                if self.is_chinese(word) is False:
                    continue
                re_list.append(word)
            self.remove_corpus.append("".join(re_list))
        print("清理完成.")

    def write(self, list, path):
        print("正在写入新文件......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for line in list:
            file.writelines(line + "\n")
        file.close()
        print("写入完成")

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False


if __name__ == "__main__":
    print("清理英文,日文,德语,中文标点,乱码等一些字符")

    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    (options, args) = parser.parse_args()

    input = options.input
    output = options.output

    try:
        Clean(infile=input, outfile=output)
        print("全部完成.")
    except Exception as err:
        print(err)