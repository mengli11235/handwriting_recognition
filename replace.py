# coding: utf-8


def rreplace(self, old, new, *max):
    count = len(self)
    if max and str(max[0]).isdigit():
        count = max[0]
    return new.join(self.rsplit(old, count))


# 这是一个逆序输出的函数，我在网上看到的，但我想了一下，因为我们现有的输出已经逆序了
# 但是词内顺序又是顺着来的，比如alef，所以不用逆序
# 放在这里仅供参考即可
# https://blog.csdn.net/c465869935/article/details/71106967（详解在这里）

f = open(r'./P25-Fg001.txt')
text = f.read()

a = text.replace("Alef", "א")
b = a.replace("Ayin", "ע")
c = b.replace("Bet", "ב")
d = c.replace("Dalet", "ד")
e = d.replace("Gimel", "ג")
f = e.replace("He", "ה")
g = f.replace("Het", "ח")
h = g.replace("Kaf-final", "ך")
i = h.replace("Kaf", "כ")
j = i.replace("Lamed", "ל")
k = j.replace("Mem-medial", "מ")
l = k.replace("Mem", "ם")
m = l.replace("Nun-final", "ן")
n = m.replace("Nun-medial", "נ")
o = n.replace("Pe-final", "ף")
p = o.replace("Pe", "פ")
q = p.replace("Qof", "ק")
r = q.replace("Resh", "ר")
s = r.replace("Samekh", "ס")
t = s.replace("Shin", "ש")
u = t.replace("Taw", "ת")
v = u.replace("Tet", "ט")
w = v.replace("Tsadi-final", "ץ")
x = w.replace("Tsadi-medial", "צ")
y = x.replace("Waw", "ו")
z = y.replace("Yod", "י")
result = z.replace("Zayin", "ז")

fo = open(r'./P25-Fg001_new.txt', 'w', encoding='utf-8')
fo.write(result)
fo.close()
