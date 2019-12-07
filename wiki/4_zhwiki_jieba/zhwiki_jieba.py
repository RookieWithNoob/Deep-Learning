import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
from optparse import OptionParser
import sys

def cut_words(sentence):
    return " ".join(jieba.cut(sentence))


if __name__ == "__main__":
    print("开始进行分词")
    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="traditional file")
    parser.add_option("--output", dest="output", default="", help="simplified file")
    (options, args) = parser.parse_args()

    input = options.input
    output = options.output

    # f = codecs.open(input, "r", encoding="utf-8")
    # target = codecs.open(output, "w", encoding="utf-8")
    print("打开文件")
    with codecs.open(input, "r", encoding="utf-8") as f:
        line_num = 1
        line = f.readline()
        while line:
            # 读取到最后一行没有内容才会结束循环
            if line_num % 1000 == 0:
                sys.stdout.write("\r--- processing  {}  article ---".format(line_num))
            # print("--- processing  ", line_num, "  article ---")
            line_seg = cut_words(sentence=line)
            with codecs.open(output, "w", encoding="utf-8") as target:
                target.writelines(line_seg)
            line_num += 1
            line = f.readline()




