import numpy as np
mat1=np.zeros((3,4))
print(mat1)

mat2=np.ones((4,5))
print(mat2)

mat3=mat2+3
print(mat3)

mat4=mat3-1
mat5=mat4*mat3
print(mat5)

mat6=np.eye(3,3)
print(mat6)

mat7=np.random.rand(3)
print(mat7)

mat8=np.random.randn(3,4)
print(mat8)

mat9=np.full([3,4],20)
print(mat9)

mat10=np.arange(2,20,4)
print(mat10)

mat11=np.linspace(2,20,5)
print(mat11)

