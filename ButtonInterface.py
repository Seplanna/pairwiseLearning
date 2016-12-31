import Tkinter as tk
#from tk import *
from PIL import Image, ImageTk
from Play import *
import os, sys

def Show(f, f1, old_label_image, root):
    image1 = Image.open(f)
    image1.resize((image1.size[0] / 2, image1.size[1] / 2), Image.ANTIALIAS)
    image2 = Image.open(f1)
    image2.resize((image2.size[0] / 2, image2.size[1] / 2), Image.ANTIALIAS)
    root.geometry('%dx%d' % (2 * image1.size[0], image1.size[1] + 80))
    tkpi = ImageTk.PhotoImage(image1)
    label_image = tk.Label(root, image=tkpi)
    label_image.place(x=0, y=0, width=image1.size[0], height=image1.size[1])
    tkpi2 = ImageTk.PhotoImage(image2)
    label_image2 = tk.Label(root, image=tkpi2)
    label_image2.place(x=image1.size[0], y=0, width=image1.size[0], height=image1.size[1])
    root.title(f)
    if old_label_image is not None:
        old_label_image.destroy()
    old_label_image = label_image
    root.mainloop()
    print("AfterMainloop")
    return  old_label_image

class ShowImages(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.items_names = GetItemsNames("../dataset/ml-1m/movies_mine.dat")
        self.items_test = np.genfromtxt("data0/test_items.txt")
        #self.my_questions = np.genfromtxt("data/quest_item_random")#"data/questions_items")
        self.my_questions = np.genfromtxt("data0/questions_items")
        self.step = 0
        self.item_vecs, self.item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData("data0")
        self.inverse_matrix = np.linalg.inv(np.eye(self.item_vecs.shape[1]) * 0.001)
        self.answers = []
        self.used_items = []
        self.questions = []
        self.user_estim = np.mean(user_vecs_train, axis=0)

            # create a prompt, an input box, an output label,
            # and a button to do the computation
        self.old_label = None
        self.Last_answer = None
        #self.prompt = tk.Label(self, text="Enter a number:", anchor="w")
        self.show_images()
        self.submit1 = tk.Button(self, text="Img1", command=lambda:self.calculate(1))
        self.submit2 = tk.Button(self, text="Img2", command=lambda:self.calculate(-1))
        self.submit3 = tk.Button(self, text="Do not know", command=lambda:self.calculate(0))
        self.entry = tk.Entry(self)
        self.output = tk.Label(self, text="")

        # lay the widgets out on the screen.
        #self.prompt.pack(side="top", fill="x")
        #self.entry.pack(side="bottom", fill="x", padx=20)
        self.output.pack(side="top", fill="x", expand=True)
        self.submit1.pack(side="left")
        self.submit2.pack(side="right")
        self.submit3.pack()

    def get_posters(self, item, comparative_item):
        item_name = self.items_names[self.items_test[item] + 1][0]
        item_name = item_name.split('?')[-1].split('(')[0]
        comparative_item_name = self.items_names[self.items_test[comparative_item] + 1][0]
        comparative_item_name = comparative_item_name.split('?')[-1].split('(')[0]
        return "images/" + item_name + ".jpg", "images/" + comparative_item_name + ".jpg"

    def show_images(self):
        win = False
        while (not win):
            try:
                if (len(self.answers) > -1):
                    self.item, self.comparative_item, self.used_items = \
                    OneIteration(self.item_vecs, self.item_bias, self.user_estim,
                         self.used_items, self.questions, self.answers, True)
                    print(self.item, win)
                else:
                    self.item = int(self.my_questions[self.step][0])
                    self.comparative_item = int(self.my_questions[self.step][1])
                    self.used_items.append(self.item)
                    self.used_items.append(self.comparative_item)
                    self.step += 1
                    print (self.step, self.comparative_item, self.item)
                print(self.items_names)
                print(self.items_names[self.items_test[self.item] + 1][0],
                      self.items_names[self.items_test[self.item] + 1][-1],
                      self.items_names[self.items_test[self.comparative_item] + 1][0],
                      self.items_names[self.items_test[self.comparative_item] + 1][-1])
                self.img1, self.img2 = self.get_posters(self.item, self.comparative_item)
                #self.img1, self.img2 = Get_posters(self.items_test, self.items_names,
                #                                   self.item, self.comparative_item)
                #self.old_label = Show(self.img1, self.img2, self.old_label, root)

                #show pictures
                print(self.img1, self.img2)
                image1 = Image.open(self.img1)
                image2 = Image.open(self.img2)
                win = True
                print(win)
                #self.image1 = tk.PhotoImage(self.img1)
                #self.image2 = tk.PhotoImage(self.img2)

            except:
                win = False
        image1.resize((image1.size[0] / 2, image1.size[1] / 2), Image.ANTIALIAS)
        image2.resize((image2.size[0] / 2, image2.size[1] / 2), Image.ANTIALIAS)
        root.geometry('%dx%d' % (2 * image1.size[0], image1.size[1] + 80))
        self.tkpi = ImageTk.PhotoImage(image1)
        label_image = tk.Label(root, image=self.tkpi)
        label_image.place(x=0, y=0, width=image1.size[0], height=image1.size[1])
        self.tkpi2 = ImageTk.PhotoImage(image2)
        label_image2 = tk.Label(root, image=self.tkpi2)
        label_image2.place(x=image1.size[0], y=0, width=image1.size[0], height=image1.size[1])
        root.title(self.img1)
        if self.old_label is not None:
            self.old_label.destroy()
        self.ld_label_image = label_image
        if (len(self.answers) > 20):
            for i in range(10):
                recom_item = BestItem(self.item_vecs, self.user_estim, self.used_items, self.item_bias, 0.2)
                self.used_items.append(recom_item)
                print(self.items_names[self.items_test[recom_item] + 1][0].split('?')[-1].split('('))
        movie_title = self.items_names[self.items_test[self.item] + 1][0].split('?')[-1].split('(')[0]
        movie_title1 = self.items_names[self.items_test[self.comparative_item] + 1][0].split('?')[-1].split('(')[0]

        #os.remove(self.img1)
        #os.remove(self.img2)
        #os.move(self.img1, "images/" + movie_title + ".jpg")
        #os.move(self.img2, "images/" + movie_title + ".jpg")

    def calculate(self, answer):

        self.user_answer = float(answer)#float(self.entry.get())
        print(self.user_answer)
        if (answer != -10):
            self.answers, self.inverse_matrix, self.questions = \
            Update(self.answers, self.user_answer,
               self.item_vecs, self.item_bias, self.item, self.comparative_item,
               self.questions)
            self.user_estim = np.dot(self.inverse_matrix,
                            np.dot(np.array(self.answers),
                                   np.array(self.questions)))
        self.show_images()





        #user_estim, answers, inverse_matrix, used_items, questions = \
        #OneIteration(item_vecs, item_bias, user_estim, used_items, questions, answers, items_names, items_test)

        #print(self.user_estim)
        #root.mainloop()


class Example(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        # create a prompt, an input box, an output label,
        # and a button to do the computation
        self.prompt = tk.Label(self, text="Enter a number:", anchor="w")
        self.entry = tk.Entry(self)
        self.submit = tk.Button(self, text="Submit", command = self.calculate)
        self.output = tk.Label(self, text="")

        # lay the widgets out on the screen.
        self.prompt.pack(side="top", fill="x")
        self.entry.pack(side="top", fill="x", padx=20)
        self.output.pack(side="top", fill="x", expand=True)
        self.submit.pack(side="right")

    def calculate(self):
        # get the value from the input widget, convert
        # it to an int, and do a calculation
        try:
            i = int(self.entry.get())
            result = "%s*2=%s" % (i, i*2)
        except ValueError:
            result = "Please enter digits only"

        # set the output widget to have our result
        self.output.configure(text=result)

# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('+%d+%d' % (100, 100))
    ShowImages(root).pack(fill="both", expand=True)
    root.mainloop()


"""
import os, sys
import Tkinter

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

root = Tkinter.Tk()
root.bind("<Button>", button_click_exit_mainloop)
root.geometry('+%d+%d' % (100,100))
dirlist = os.listdir('.')
old_label_image = None
for f in dirlist:
    try:
        image1 = Image.open("1.jpg")
        root.geometry('%dx%d' % (image1.size[0],image1.size[1]))
        tkpi = ImageTk.PhotoImage(image1)
        label_image = Tkinter.Label(root, image=tkpi)
        label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
        root.title(f)
        if old_label_image is not None:
            old_label_image.destroy()
        old_label_image = label_image
        root.mainloop() # wait until user clicks the window
    except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        pass
        """
"""def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

if __name__ == "__main__":
    old_label_image = None
    root = tk.Tk()
    #root.bind("<Button>", button_click_exit_mainloop)
    #root.geometry('+%d+%d' % (100, 100))
    #old_label_image = Show("1.jpg","1.jpg", old_label_image, root)
    Example(root).pack(fill="both", expand=True)"""


