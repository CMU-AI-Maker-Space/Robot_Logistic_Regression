import numpy as np
import matplotlib.backend_bases as backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data point attributes and related labels
X = []
y = []
# Create Learning model
# TODO: You can set some of its hyperparameters if you with. Take a look at the documentation
model = LogisticRegression()

def gui_setup():
    # Start figure and define its limits
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    # Figure labels and appearance
    ax.set_title("Chance of robot failure")
    ax.set_xlabel("Operation time (h)")
    ax.set_ylabel("Distance covered (km)")
    ax.grid()
    # Attach click function
    fig.canvas.mpl_connect('button_press_event', on_click)
    # Return the figure and the axes for future use
    return fig, ax

def on_click(event: backend.MouseEvent):
    # Only do anything if the click was inside the valid area
    ax = event.inaxes
    if ax is not None:
        data_point = [event.xdata, event.ydata]

        # Left click is a failure
        if event.button == backend.MouseButton.LEFT:
            label = 0
            # Insert the new point in the saved data points
            X.append(data_point)
            y.append(label)

        # Right click is a non-failure
        elif event.button == backend.MouseButton.RIGHT:
            label = 1
            # Insert the new point in the saved data points
            X.append(data_point)
            y.append(label)

        # Middle click is clear
        elif event.button == backend.MouseButton.MIDDLE:
            X.clear()
            y.clear()

        # If some other button was clicked, just do nothing
        else:
            return

        # Update the image with all points
        update_plot()
        # Redraw the figure
        plt.draw()

def update_plot():
    # Clear points
    for artist in plt.gca().collections + plt.gca().lines:
        artist.remove()

    # Transforming the data to ndarray as it is easier to work with
    data_points = np.array(X)
    labels = np.array(y)

    failure_points = data_points[np.where(labels == 0)[0]]
    success_points = data_points[np.where(labels == 1)[0]]

    if len(failure_points) > 0:
        plt.scatter(failure_points[:, 0], failure_points[:, 1], marker='x', c='red')
    if len(success_points) > 0:
        plt.scatter(success_points[:, 0], success_points[:, 1], marker='o', c='blue')

    # If we have both successes and failures
    # fit a logistic regression model to the data
    # NOTE: This doce does not take test error into account. Can you implement it?
    if len(failure_points) > 0 and len(success_points) > 0:
        (x_coef, y_coef, intercept_coef), acc = fit_logistic(data_points, labels)
        # Line is just two points
        x_pts = [-1000, 1000]
        y_pts = []
        for x in x_pts:
            y_pts.append(-(intercept_coef+x*x_coef)/y_coef)
        plt.plot(x_pts, y_pts, c='green')
        # Inform the user of the accuracy
        print(f'Current training accuracy: {100*acc:.2f}%')
        print()
    

def fit_logistic(data_points: np.ndarray, labels: np.ndarray):
    # Train the model on the data
    model.fit(data_points, labels)
    # Find the mean accuracy on the training data
    acc = model.score(data_points, labels)
    # Return the 50% line data from training and the accuracy on the training data
    return (model.coef_[0, 0], model.coef_[0, 1], model.intercept_[0]), acc

def main():
    gui_setup()
    # User instructions
    print("Populate the screen with data :-)")
    print("Left click for FAILURE")
    print("Right click for NON FAILURE")
    print("Middle button to CLEAR")
    plt.show()

if __name__ == "__main__":
    main()
