import matplotlib.pyplot as plt

# 1 (a)

data = [
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
    (-1, -1, -1)
]

x1 = [point[0] for point in data]
x2 = [point[1] for point in data]
labels = [point[2] for point in data]

colors = ['red' if label == 1 else 'blue' for label in labels]

plt.scatter(x1, x2, c=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Boolean Function Example')
plt.savefig("images/boolean_function_example.png")
plt.show()
