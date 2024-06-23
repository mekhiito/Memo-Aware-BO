a = list()

a.append([])
a.append([])
a.append([1,2])
a.append([[1,2]])
if [[1,2]] not in a:
    a.append('NOT HERE')
else:
    a.append('HERE')
print(a)