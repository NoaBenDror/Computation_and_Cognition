from hamster import myHamster
import matplotlib.pyplot as plt

id = 260
Xg = 1
u = [1]
results = [1]
for i in range(5):
    Xs = 0
    choice1 = myHamster(Xs, Xg, id)
    Xsi = choice1
    while choice1 == Xsi and Xs < Xg:
       Xs += 0.01
       Xsi = myHamster(Xs, Xg, id)

    u.append(0.5 * u[-1])
    results.append(Xs)
    Xg = Xs

u.append(0)
results.append(0)

plt.plot(results, u)
plt.xlabel("Xs")
plt.ylabel("U - value")
plt.title("Utility function - ID 260")
plt.show()