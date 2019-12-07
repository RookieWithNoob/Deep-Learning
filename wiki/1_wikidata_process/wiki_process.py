"""
    FILE :  wiki_process.py
    FUNCTION : None
    REFERENCE: https://github.com/AimeeLee77/wiki_zh_word2vec/blob/master/1_process.py
    python3 wiki_process.py zhwiki-latest-pages-articles.xml.bz2 zhwiki-latest.txt
    2019-12-05 00:22:09,467: INFO: running wiki_process.py zhwiki-latest-pages-articles.xml.bz2 zhwiki-latest.txt
    2019-12-05 00:23:20,005: INFO: Saved 10000 articles.
    2019-12-05 00:24:28,225: INFO: Saved 20000 articles.

"""

import logging
import os.path
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary=[])
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles.")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles.")
