import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
import matplotlib.animation as animation
import numpy as np

def plot_3d(data):
    """
    data = [ data0, data1, data2, .... data_n]
    data0 = [x, y, z]
    x = [a, b, c]
    :param data:
    :return:
    """
    fig = plt.figure()
    ax = plt3.Axes3D(fig)   

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    # make 3 lines - x, y, z
    lines = [ax.plot([0,data[0,i,0]], [0,data[0,i,1]], [0,data[0,i,2]])[0] for i in range(3)]
    line_ani = animation.FuncAnimation(fig, update_lines, data.shape[0], fargs=(data, lines),
                                       interval=1, blit=False)
    plt.show()

def update_lines(num, dataLines, lines):
    print(num)
    data = dataLines[num]

    for line, dat in zip(lines, data):
        x = np.array([0, dat[0]])
        y = np.array([0, dat[1]])
        z = np.array([0, dat[2]])
        line.set_data(x, y)
        line.set_3d_properties(z)

    return lines