from pybloom_live import BloomFilter

bf = BloomFilter(capacity=1000)

bf.add("www.baidu.com")



print("www.baidu.com" in bf)   # True
print("www.douban.com" in bf)  # False
