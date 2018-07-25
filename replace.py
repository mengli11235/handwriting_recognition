# coding: utf-8
import os

f = open(r'./P123-Fg002-R-C01-R01-fused.txt')
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
result = result.replace("-", "")

fo = open(r'./tmp.txt', 'w', encoding='utf-8')
fo.write(result)
fo.close()

fn = open(r'./new.txt', 'w', encoding='utf-8')

for line in list(open(r'./tmp.txt')):
    b = line[::-1]
    fn.writelines(b.strip() + '\n')
fo.close()

os.remove('./tmp.txt')
