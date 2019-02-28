import tkinter
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
import cv2, math

from tkinter import filedialog, Button, Label, Radiobutton

flag1 = 0
window = tkinter.Tk()
window.title('Image Restoration_Add Noise')
window.geometry("750x550+102+120")
o_img, gray, image_out, cv_image_out, noise_img, cv_noise_img = None, None, None, None, None, None
x1_line_pt, y1_line_pt, x2_line_pt, y2_line_pt = None, None, None, None
flag = 0
left_but = "up"
x_pos, y_pos = None, None


class gui:

    def median_filter(self, image, window3):
        median_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                pixel.sort()
                median_image[row, col] = pixel[int(pixel.size / 2)]
        self.cv_image_out = median_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def max_filter(self, image, window3):
        max_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                max_image[row, col] = max(pixel)
        self.cv_image_out = max_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def arithmetic_filter(self, image, window3):
        val = 0
        pixels = 0
        height = image.shape[0]
        width = image.shape[1]
        arithmetic_image = image.copy()
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                total = sum(pixel)
                pixels = total / 9
                arithmetic_image[row][col] = pixels
        self.cv_image_out = arithmetic_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def geometric_filter(self, image, window3):
        height = image.shape[0]
        width = image.shape[1]
        val = 0
        total1 = 0
        # product = 1
        geo_image = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                prod = 1
                m = 1 / len(pixel)
                for p in pixel:
                    p = math.pow(p, m)
                    prod *= p
                geo_image[row][col] = int(prod)
        geo_image.astype(int)
        self.cv_image_out = geo_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def min_filter(self, image, window3):
        median_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                median_image[row, col] = min(pixel)
        self.cv_image_out = median_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def midpoint_filter(self, image, window3):
        midpoint_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                midpoint_image[row, col] = min(pixel) / 2 + max(pixel) / 2
        self.cv_image_out = midpoint_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def alpha_trimmed_mean(self, image, window3):
        alpha_trimmed_image = image.copy()
        d = 4
        m = n = 3
        b = int((m * n) - d)
        trim_factor = int(d / 2)

        height = image.shape[0]
        width = image.shape[1]
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                pixel.sort()
                pixel_size = pixel.size

                if b != 0:
                    trimmed_pixel = pixel[trim_factor:pixel_size - trim_factor]
                else:
                    trimmed_pixel = pixel

                pixel_sum = 0
                for pixel in trimmed_pixel:
                    pixel_sum += pixel

                alpha_trimmed_image[row, col] = int(pixel_sum / b)
        self.cv_image_out = alpha_trimmed_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def adaptive_filter(self, image, window3):
        adaptive_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        local_mean_matrix = np.zeros((height, width))
        local_variance_matrix = np.zeros((height, width))

        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors(row, col, image, 1)
                local_mean_matrix[row, col] = np.mean(pixel)
                local_variance_matrix[row, col] = np.var(pixel)

        variance = np.average(local_variance_matrix)
        for row in range(height):
            for col in range(width):
                if variance > local_variance_matrix[row, col]:
                    local_variance_matrix[row, col] = variance

                adaptive_image[row, col] = image[row, col] - (variance / local_variance_matrix[row, col]) * (
                            image[row, col] - local_mean_matrix[row, col])

        self.cv_image_out = adaptive_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")
        return adaptive_image

    def get_neighbors(self, row, col, img, distance):
        return img[max(row - distance, 0):min(row + distance + 1, img.shape[0]),
               max(col - distance, 0):min(col + distance + 1, img.shape[1])].flatten()

    def contra_harmonic(self, image, window3):
        contra_harmonic_image = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        q = float(2)
        for row in range(height):
            for col in range(width):
                pixel = self.get_neighbors1(row, col, width, height, image)
                s1 = [x ** q for x in pixel]
                s2 = [x ** (q + 1) for x in pixel]
                contra_harmonic_image[row, col] = sum(s2) / sum(s1)

        self.cv_image_out = contra_harmonic_image
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")
        return contra_harmonic_image

    def get_neighbors1(self, row, col, width, height, img):
        left = max(0, col - 1)
        up = max(0, row - 1)
        right = min(width - 1, col + 1)
        down = min(height - 1, row + 1)
        pixel = [img[up][left], img[up][col],
                 img[up][right], img[row][right],
                 img[row][left], img[down][left],
                 img[down][col], img[down][right]]

        return pixel

    def noisy(self, noise_typ, image, window2):
        if noise_typ == "gauss":
            row, col = image.shape
            mean = 0
            var = 2
            sigma = var ** 4
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = image + gauss
            self.cv_noise_img = noisy.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")
        elif noise_typ == "reyleigh":
            row, col = image.shape
            mean_value = 15
            mode_value = np.sqrt(2 / np.pi) * mean_value
            reyleigh = np.random.rayleigh(mode_value, (row, col))
            reyleigh = reyleigh.reshape(row, col)
            noisy = image + reyleigh
            self.cv_noise_img = noisy.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")
        elif noise_typ == "uniform":
            row, col = image.shape
            uniform = np.random.uniform(-10, 50, (row, col))
            uniform = uniform.reshape(row, col)
            noisy = image + uniform
            self.cv_noise_img = noisy.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")
        elif noise_typ == "salt":
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 255
            self.cv_noise_img = out.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")
        elif noise_typ == "pepper":
            # Pepper mode
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            self.cv_noise_img = out.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")
        elif noise_typ == "sp":
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            self.cv_noise_img = out.astype('uint8')
            self.noise_img = Image.fromarray(self.cv_noise_img)
            self.noise_img = ImageTk.PhotoImage(self.noise_img.resize((200, 200)))
            Label(window2, image=self.noise_img).place(relx=.71, rely=.35, anchor="c")

    def harmonic_mean_filter(self, img, mask_size, window3):
        #        global image_out
        def harmonic_mean(numbers):
            reciprocals = []
            for i in numbers:
                reciprocals.append(1 / (i + 1))
            average = sum(reciprocals) / len(numbers)
            hm_decimal = 1 / average
            return round(hm_decimal)

        img_out = img.copy()
        height = img.shape[0]
        width = img.shape[1]

        for i in range(height):
            for j in range(width):
                neighbors = []
                for k in np.arange(-int(mask_size / 2), int(mask_size / 2)):
                    for l in np.arange(-int(mask_size / 2), int(mask_size / 2)):
                        if (i + k) >= 0 and (i + k) < height and (j + l) >= 0 and (j + l) < width:
                            a = img.item(i + k, j + l)
                            neighbors.append(a)
                b = harmonic_mean(neighbors)
                img_out.itemset((i, j), b)
        self.cv_image_out = img_out
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((200, 200)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def browsefunc(self, window):
        filename = filedialog.askopenfilename()

        image = cv2.imread(filename)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.o_img = Image.fromarray(self.gray)
        self.o_img = ImageTk.PhotoImage(self.o_img.resize((200, 200)))
        Label(window, image=self.o_img).place(relx=.5, rely=.3, anchor="c")

    def rectangle_draw(self, event=None):
        #        global flag, x1_line_pt, y1_line_pt, x2_line_pt, y2_line_pt

        if self.flag1 == 0:

            if None not in (self.x1_line_pt, self.y1_line_pt, self.x2_line_pt, self.y2_line_pt):
                # fill : Color option names are here http://wiki.tcl.tk/37701
                # outline : border color
                # width : width of border in pixels

                event.widget.create_rectangle(self.x1_line_pt, self.y1_line_pt, self.x2_line_pt, self.y2_line_pt,
                                              fill="", outline="black", width=2)
                print(self.x1_line_pt, "..", self.y1_line_pt, "...", self.x2_line_pt, "...", self.y2_line_pt)
                self.flag1 = 1

    def left_but_down(self, event=None):
        #        global left_but,x1_line_pt,y1_line_pt
        self.left_but = "down"

        # Set x & y when mouse is clicked
        self.x1_line_pt = event.x
        self.y1_line_pt = event.y

    # ---------- CATCH MOUSE UP ----------

    def left_but_up(self, event=None):
        #        global x2_line_pt,y2_line_pt, x_pos,y_pos,left_but
        self.left_but = "up"

        # Reset the line
        self.x_pos = None
        self.y_pos = None

        # Set x & y when mouse is released
        self.x2_line_pt = event.x
        self.y2_line_pt = event.y

        # If mouse is released and line tool is selected
        # draw the line

        self.rectangle_draw(event)

    def image(self, pil_image):

        root = tkinter.Toplevel()
        w, h = pil_image.width(), pil_image.height()
        drawing_area = Canvas(root, height=h, width=w)
        self.flag1 = 0
        drawing_area.create_image(0, 0, image=pil_image, anchor="nw")
        drawing_area.pack()
        drawing_area.bind("<ButtonPress-1>", self.left_but_down)
        drawing_area.bind("<ButtonRelease-1>", self.left_but_up)

    def page5(self, window4):

        window4.withdraw()
        window5 = tkinter.Toplevel()
        window5.title('Image Restoration_Add Noise')
        window5.geometry("800x680+102+120")
        n_graph = Label(window5, text="Noise Graph Image")

        n_graph.place(relx=.1, rely=.1, anchor="c")
        #        cv2.imshow("img",self.cv_noise_img[self.y2_line_pt:self.y1_line_pt,self.x1_line_pt:self.x2_line_pt])

        plt.figure(0)
        plt.hist(self.cv_noise_img[self.y1_line_pt:self.y2_line_pt, self.x1_line_pt:self.x2_line_pt].ravel(), 256,
                 [0, 256])
        plt.savefig("noiseplot.png")

        self.noised = cv2.imread("noiseplot.png")
        self.noised = Image.fromarray(self.noised)
        self.noised = ImageTk.PhotoImage(self.noised.resize((700, 250)))

        Label(window5, image=self.noised).place(relx=.42, rely=.3, anchor="c")


        r_graph = Label(window5, text="Restored Graph Image")
        r_graph.place(relx=.1, rely=.5, anchor="c")

        plt.figure(1)
        plt.hist(self.cv_image_out[self.y1_line_pt:self.y2_line_pt, self.x1_line_pt:self.x2_line_pt].ravel(), 256,
                 [0, 256])
        plt.savefig("restoreplot.png")

        self.restored = cv2.imread("restoreplot.png")
        self.restored = Image.fromarray(self.restored)
        self.restored = ImageTk.PhotoImage(self.restored.resize((700, 250)))

        Label(window5, image=self.restored).place(relx=.42, rely=.7, anchor="c")

        Button(window5, text="Next Image", command=lambda: self.page1(window5)).place(relx=.4, rely=.95, anchor="c")
        Button(window5, text="Quit", command=window5.destroy).place(relx=.6, rely=.95, anchor="c")

    def page4(self, window3):
        window3.withdraw()
        window4 = tkinter.Toplevel()
        window4.title('Image Restoration_Add Noise')
        window4.geometry("750x550+102+120")

        or_img4 = Label(window4, text="Original Image")
        or_img4.place(relx=.12, rely=.1, anchor="c")
        Label(window4, image=self.o_img).place(relx=.18, rely=.38, anchor="c")

        img_noise2 = Label(window4, text="Image with noise")
        img_noise2.place(relx=.48, rely=.1, anchor="c")
        Label(window4, image=self.noise_img).place(relx=.485, rely=.38, anchor="c")

        re_img = Label(window4, text="Restored Image")
        re_img.place(relx=.75, rely=.1, anchor="c")
        Label(window4, image=self.image_out).place(relx=.8, rely=.38, anchor="c")

        select_image = Button(window4, text="Select Region", command=lambda: self.image(self.noise_img))
        select_image.place(relx=.5, rely=.8, anchor="c")

        browsebutton4 = Button(window4, text="Compare Histogram", command=lambda: self.page5(window4))
        browsebutton4.place(relx=.5, rely=.9, anchor="c")

    def page3(self, window2):
        window2.withdraw()
        window3 = tkinter.Toplevel()
        window3.title('Image Restoration_Add Noise')
        window3.geometry("750x550+120+120")

        mfilter = Label(window3, text="Mean Filter")
        mfilter.place(relx=.5, rely=.06, anchor="c")
        Aritmetic = Button(window3, text="Aritmetic",
                           command=lambda: self.arithmetic_filter(self.cv_noise_img, window3))
        Aritmetic.place(relx=.5, rely=.12, anchor="c")
        Geometric = Button(window3, text="Geometric", command=lambda: self.geometric_filter(self.cv_noise_img, window3))
        Geometric.place(relx=.5, rely=.18, anchor="c")
        Harmonic = Button(window3, text="Harmonic",
                          command=lambda: self.harmonic_mean_filter(self.cv_noise_img, 3, window3))
        Harmonic.place(relx=.5, rely=.24, anchor="c")
        ContraHarmonic = Button(window3, text="ContraHarmonic",
                                command=lambda: self.contra_harmonic(self.cv_noise_img, window3))
        ContraHarmonic.place(relx=0.5, rely=.30, anchor="c")

        img_noise3 = Label(window3, text="Image with noise")
        img_noise3.place(relx=.18, rely=.1, anchor="c")
        Label(window3, image=self.noise_img).place(relx=.2, rely=.4, anchor="c")

        re_img3 = Label(window3, text="Restored Image")
        re_img3.place(relx=.8, rely=.1, anchor="c")

        mfilter = Label(window3, text="Order Static Filter")
        mfilter.place(relx=.5, rely=.36, anchor="c")
        median_b = Button(window3, text="Median", command=lambda: self.median_filter(self.cv_noise_img, window3))
        median_b.place(relx=.5, rely=.42, anchor="c")
        max_b = Button(window3, text="Max", command=lambda: self.max_filter(self.cv_noise_img, window3))
        max_b.place(relx=.5, rely=.48, anchor="c")
        min_b = Button(window3, text="Min", command=lambda: self.min_filter(self.cv_noise_img, window3))
        min_b.place(relx=.5, rely=.54, anchor="c")
        mid_b = Button(window3, text="MidPoint", command=lambda: self.midpoint_filter(self.cv_noise_img, window3))
        mid_b.place(relx=.5, rely=.60, anchor="c")
        alpha_b = Button(window3, text="Alpha Trimmed",
                         command=lambda: self.alpha_trimmed_mean(self.cv_noise_img, window3))
        alpha_b.place(relx=.5, rely=.66, anchor="c")

        adaptive = Label(window3, text="Adaptive Filter")
        adaptive.place(relx=.5, rely=.75, anchor="c")
        adaptive_b = Button(window3, text="Adaptive", command=lambda: self.adaptive_filter(self.cv_noise_img, window3))
        adaptive_b.place(relx=.5, rely=.81, anchor="c")

        compare = Button(window3, text="Compare Image", command=lambda: self.page4(window3))
        compare.place(relx=.5, rely=.95, anchor="c")

    def page2(self, window):
        window.withdraw()
        window2 = tkinter.Toplevel()
        window2.title('Image Restoration_Add Noise')
        window2.geometry("750x550+102+120")
        ori_img2 = Label(window2, text="Original Image")
        ori_img2.place(relx=.25, rely=.1, anchor="c")
        Label(window2, image=self.o_img).place(relx=.25, rely=.35, anchor="c")

        img_noise2 = Label(window2, text="Image with Noise")
        img_noise2.place(relx=.71, rely=.1, anchor="c")

        Radiobutton(window2, text="Salt", value=5, command=lambda: self.noisy("salt", self.gray, window2)).place(
            relx=.12, rely=.7, anchor="c")
        Radiobutton(window2, text="Pepper", value=6, command=lambda: self.noisy("pepper", self.gray, window2)).place(
            relx=.24, rely=.7, anchor="c")
        Radiobutton(window2, text="Salt & Pepper", value=1, command=lambda: self.noisy("sp", self.gray, window2)).place(
            relx=.40, rely=.7, anchor="c")
        Radiobutton(window2, text="Gaussian", value=2, command=lambda: self.noisy("gauss", self.gray, window2)).place(
            relx=.57, rely=.7, anchor="c")
        Radiobutton(window2, text="Uniform", value=3, command=lambda: self.noisy("uniform", self.gray, window2)).place(
            relx=.72, rely=.7, anchor="c")
        Radiobutton(window2, text="Reyleigh", value=4,
                    command=lambda: self.noisy("reyleigh", self.gray, window2)).place(relx=.85, rely=.7, anchor="c")
        restoring = Button(window2, text="Restoring Process", command=lambda: self.page3(window2))
        restoring.place(relx=.5, rely=.85, anchor="c")

    def page1(self, window):
        global flag
        if flag == 0:
            browsebutton = Button(window, text="Browse", command=lambda: self.browsefunc(window))
            browsebutton.place(relx=.2, rely=.8, anchor="c")

            noise = Button(window, text="Add Noise", command=lambda: self.page2(window))
            noise.place(relx=.7, rely=.8, anchor="c")
        else:
            window.withdraw()
            window_new = tkinter.Toplevel()
            window_new.title('Image Restoration_Add Noise')
            window_new.geometry("750x550+102+120")
            browsebutton = Button(window_new, text="Browse", command=lambda: self.browsefunc(window_new))
            browsebutton.place(relx=.2, rely=.8, anchor="c")

            noise = Button(window_new, text="Add Noise", command=lambda: self.page2(window_new))
            noise.place(relx=.7, rely=.8, anchor="c")


g = gui()
g.page1(window)
window.mainloop()