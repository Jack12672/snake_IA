from tkinter import * 
import random
import time
import tensorflow as tf
import numpy as np

SPEED=100
CASE=20

class Snake():
    def __init__(self,dimension):
        self.isdead=False
        self.dx=1
        self.dy=0
        self.position=[]
        self.length=3
        x1=int(dimension[0]/2)
        y1=int(dimension[1]/2)
        self.head=((x1,y1))
        self.position.append(self.head)
        y1+=1
        self.position.append((x1,y1))
        x1-=1
        self.position.append((x1,y1))
        self.fruit=((y1-3,x1+2))
        self.count=0

class Grid():
    def __init__(self,dimension:tuple [int,int]):
        self.grid=np.zeros((dimension[1],dimension[0]))
        self.grid[:, 0] = 1
        self.grid[:, dimension[0]-1] = 1
        self.grid[0, :] = 1
        self.grid[dimension[1]-1, :] = 1

        self.dimension=dimension 

class Can(Canvas):
    def __init__(self,canvas,dimension):
        self.dimension=dimension
        super().__init__(canvas,bg="white",width=dimension[0]*CASE,height=dimension[1]*CASE)
        self.obj=[]
        self.fruit=0
        self.grid=Grid(dimension)
        self.snake=Snake(dimension)
        self.god=np.zeros((1,6), dtype=np.float32)
        self.snakeIA=np.zeros((1,11),dtype=np.float32)
        self.init_grid()
        self.model=0
        self.learn=True
        self.predict=False
        self.save= False


        new_model=False

        if new_model:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Input(shape=(11,)))
            self.model.add(tf.keras.layers.Dense(128, activation='relu'))
            self.model.add(tf.keras.layers.Dense(64, activation='relu'))
            self.model.add(tf.keras.layers.Dense(3,activation='softmax'))

            self.god = tf.keras.models.Sequential()
            self.god.add(tf.keras.layers.Input(shape=(6,)))
            self.god.add(tf.keras.layers.Dense(16, activation='relu'))
            self.god.add(tf.keras.layers.Dense(8, activation='relu'))
            self.god.add(tf.keras.layers.Dense(5,activation='softmax'))

        else: 
            self.model = tf.keras.models.load_model ('SNAKE.keras')
            self.god = tf.keras.models.load_model ('GOD.keras')






    def direction(self):
        res=np.zeros((1,5))
        if self.snake.dx==0 and self.snake.dy==0:
            res[0,0]=1
        if self.snake.dx==0:
            res[0,1]=0
            res[0,2]=0
        if self.snake.dy==0:
            res[0,3]=0
            res[0,4]=0
        if self.snake.dx==-1:
            res[0,1]=1
            res[0,2]=0
        elif self.snake.dx==1:
            res[0,1]=0
            res[0,2]=1
        if self.snake.dy==-1:
            res[0,3]=1
            res[0,4]=0
        elif self.snake.dy==1:
            res[0,3]=0
            res[0,4]=1
        return res

    def init_grid(self):
        self.create_rectangle(      0, 0, 
                                                    self.dimension[0]*CASE, CASE, 
                                                    fill='orange', outline="")
        self.create_rectangle(      0, (self.dimension[1]-1)*CASE, 
                                                    self.dimension[0]*CASE, (self.dimension[1]-1)*CASE+CASE, 
                                                    fill='orange', outline="")

        self.create_rectangle(      0, 0, 
                                                    CASE, (self.dimension[0]-1)*CASE, 
                                                    fill='orange', outline="")
        self.create_rectangle(      (self.dimension[0]-1)*CASE, 0, 
                                                    (self.dimension[0]-1)*CASE+CASE, self.dimension[1]*CASE, 
                                                    fill='orange', outline="")

        for i in range (self.snake.length):
            x,y=self.snake.position[i]
            self.grid.grid[y,x]=1
            x,y=x*CASE,y*CASE
            self.obj.append(self.create_oval(x,y,x+CASE,y+CASE,width = 1, fill="green"))
        
        self.draw_fruit()
        self.draw_grid
        self.pack()

    def draw_grid(self):      
        color='blue'
        w=self.dimension[0]*CASE
        h=self.dimension[1]*CASE
        for x in range (0,w+1,CASE):
            self.create_line(x,0,x,h,width = 1, fill=color)
        for y in range (0,h+1,CASE):
            self.create_line(0,y,w,y,width = 1, fill=color)
        self.pack()

    def draw_snake(self,growth):
        if growth!=True:
            l=self.snake.length
            x,y=self.snake.position[l-1][0],self.snake.position[l-1][1]
            self.grid.grid[y,x]=0
        for i in range (self.snake.length-2,-1,-1):
            x,y=self.snake.position[i][0],self.snake.position[i][1]
            x1,y1=self.snake.position[i+1][0],self.snake.position[i+1][1]
            self.snake.position[i+1]=(x,y)
            self.move(self.obj[i+1],(x-x1)*CASE,(y-y1)*CASE)
        x,y=self.snake.head
        x1,y1=self.snake.position[0][0],self.snake.position[0][1]
        self.snake.position[0]=(x,y)
        self.grid.grid[y,x]=1
        self.move(self.obj[0],(x-x1)*CASE,(y-y1)*CASE)
        if self.snake.count>15:
            self.snake.count-=1
            self.itemconfigure(self.obj[0],fill='cyan')
        elif self.snake.count>10:
            self.snake.count-=1
            self.itemconfigure(self.obj[0],fill='deep sky blue')
        elif self.snake.count>5:
            self.snake.count-=1
            self.itemconfigure(self.obj[0],fill='DodgerBlue2')
        elif self.snake.count>0:
            self.itemconfigure(self.obj[0],fill='green')

        self.pack()

    def draw_fruit(self):
        maxx,maxy=self.dimension
        isvalid=True
        x,y=0,0     
        while isvalid:
            seed = time.time() / 3600
            random.seed(seed)
            x=random.randrange(maxx)
            seed = time.time() / 3600 
            random.seed(seed)
            y=random.randrange(maxy)
            if self.grid.grid[y,x]==0 : isvalid=False

        self.snake.fruit=(x,y)
        x0,y0=x*CASE,y*CASE
        x1,y1=x0+CASE,y0+CASE
        self.fruit=self.create_oval(x0,y0,x1,y1,width = 1, fill="red")

    def get_left(self):
        dx,dy=0,0
        if self.snake.dx==0 and self.snake.dy==1:
            dx=1
            dy=0
        elif self.snake.dx==0 and self.snake.dy==-1:
            dx=-1
            dy=0
        elif self.snake.dx==1 and self.snake.dy==0:
            dx=0
            dy=-1
        elif self.snake.dx==-1 and self.snake.dy==0:
            dx=0
            dy=1

        return (dx,dy)

    def get_right(self):
        dx,dy=0,0
        if self.snake.dx==0 and self.snake.dy==1:
            dx=-1
            dy=0
        elif self.snake.dx==0 and self.snake.dy==-1:
            dx=1
            dy=0
        elif self.snake.dx==1 and self.snake.dy==0:
            dx=0
            dy=1
        elif self.snake.dx==-1 and self.snake.dy==0:
            dx=0
            dy=-1
        return (dx,dy)

    def leff_IA(self):
        self.snake.dx,self.snake.dy=self.get_left()
        # if self.snake.dx==0 and self.snake.dy==1:
        #     self.snake.dx=1
        #     self.snake.dy=0
        # elif self.snake.dx==0 and self.snake.dy==-1:
        #     self.snake.dx=-1
        #     self.snake.dy=0
        # elif self.snake.dx==1 and self.snake.dy==0:
        #     self.snake.dx=0
        #     self.snake.dy=-1
        # elif self.snake.dx==-1 and self.snake.dy==0:
        #     self.snake.dx=0
        #     self.snake.dy=1

    def right_IA(self):
        self.snake.dx,self.snake.dy=self.get_right()
        # if self.snake.dx==0 and self.snake.dy==1:
        #     self.snake.dx=-1
        #     self.snake.dy=0
        # elif self.snake.dx==0 and self.snake.dy==-1:
        #     self.snake.dx=1
        #     self.snake.dy=0
        # elif self.snake.dx==1 and self.snake.dy==0:
        #     self.snake.dx=0
        #     self.snake.dy=1
        # elif self.snake.dx==-1 and self.snake.dy==0:
        #     self.snake.dx=0
        #     self.snake.dy=-1


    def update_IA(self):
        x,y = self.snake.head
        dx,dy=self.snake.dx,self.snake.dy
        xf,yf = self.snake.fruit

        #test direction
        if dx==-1 : 
            self.snakeIA[0,3]=1
            self.snakeIA[0,4]=0
            self.snakeIA[0,5]=0
            self.snakeIA[0,6]=0
        elif dx==1 : 
            self.snakeIA[0,3]=0
            self.snakeIA[0,4]=1
            self.snakeIA[0,5]=0
            self.snakeIA[0,6]=0
        elif dy==-1 : 
            self.snakeIA[0,3]=0
            self.snakeIA[0,4]=0
            self.snakeIA[0,5]=1
            self.snakeIA[0,6]=0
        elif dy==1 : 
            self.snakeIA[0,3]=0
            self.snakeIA[0,4]=0
            self.snakeIA[0,5]=0
            self.snakeIA[0,6]=1

        # tests obstacles       
        x_front,y_front=x+dx,y+dy #obstacle devant
        if x_front>=self.dimension[0] or y_front>=self.dimension[1]:
            self.snakeIA[0,0]=1  
        elif self.grid.grid[y_front,x_front]==1: 
            self.snakeIA[0,0]=1
        else: self.snakeIA[0,0]=0
        
        dx,dy=self.get_left()#obstacle gauche
        x_left,y_left=x+dx,y+dy
        if x_left>=self.dimension[0] or y_left>=self.dimension[1]:
            self.snakeIA[0,1]=1  
        elif self.grid.grid[y_left,x_left]==1: 
            self.snakeIA[0,1]=1
        else: self.snakeIA[0,1]=0

        dx,dy=self.get_right() #obstacle droite
        x_rignt,y_right=x+dx,y+dy
        if x_rignt>=self.dimension[0] or y_right>=self.dimension[1]:
            self.snakeIA[0,2]=1  
        elif self.grid.grid[y_right,x_rignt]==1:
            self.snakeIA[0,2]=1
        else: self.snakeIA[0,2]=0

        #test position fruit
        if xf<x:
            self.snakeIA[0,7]=1
            self.snakeIA[0,8]=0
        elif xf>x:
            self.snakeIA[0,7]=0
            self.snakeIA[0,8]=1
        else: 
            self.snakeIA[0,7]=0
            self.snakeIA[0,8]=0
        if yf<y:
            self.snakeIA[0,9]=1
            self.snakeIA[0,10]=0
        elif yf>y:
            self.snakeIA[0,9]=0
            self.snakeIA[0,10]=1
        else: 
            self.snakeIA[0,9]=0
            self.snakeIA[0,10]=0
 
        print(self.snakeIA)

        if self.learn:
            score=np.zeros((1,3),dtype=np.float32)
            # test mange fruit
            


    def update(self):
            self.update_IA()
            x,y=self.snake.position[0][0]+self.snake.dx,self.snake.position[0][1]+self.snake.dy
            self.snake.head=(x,y)
            growth=False
            
            if self.check_obstacle():
                print('end')
            if self.check_fruit():
                growth=True
            if self.snake.isdead==False:
                self.draw_snake(growth)


    def check_obstacle (self):
        x,y = self.snake.head
        res=False
        if x<0 or x>self.grid.dimension[0]-1 : 
            self.snake.isdead=True
            res=True
        if y<0 or y>self.grid.dimension[1]-1: 
            self.snake.isdead=True
            res=True
        if self.snake.isdead==False and self.grid.grid[y,x]!=0:
            res=True
            self.snake.isdead=True
        return res


    def check_fruit(self):
        x,y = self.snake.head
        xf,yf = self.snake.fruit
        if (x==xf and y==yf): 
            self.delete(self.fruit)
            l=self.snake.length
            x,y=self.snake.position[l-1][0],self.snake.position[l-1][1]
            self.snake.position.append((x,y))
            x,y=x*CASE,y*CASE
            self.obj.append(self.create_oval(x,y,x+CASE,y+CASE,width = 1, fill="green"))
            self.snake.length+=1
            self.snake.count=20
            self.draw_fruit()
            return True
        else: return False





class Window_0(Frame):
    def __init__(self,dimension): 
        Frame.__init__(self)
        self.w=Can(self.master,dimension)
        w=dimension[0]*CASE
        h=dimension[1]*CASE
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        posx=int((screen_width-w)/2)
        posy=int((screen_height-h)/2)
        self.master.title('Snake IA 2025')
        self.master.geometry('%dx%d+%d+%d' % (w,h,posx,posy))
        self.master.bind("<KeyPress-Left>",lambda event: self.left(event))
        self.master.bind("<KeyPress-Right>",lambda event: self.right(event))
        self.master.bind("<KeyPress-Up>",lambda event: self.up(event))
        self.master.bind("<KeyPress-Down>",lambda event: self.down(event))

        self.master.bind("<Button-1>",lambda event: self.mousedown_left(event))
        self.master.bind("<Button-2>",lambda event: self.mousedown_scroll_wheel(event))
        self.master.bind("<Button-3>",lambda event: self.mousedown_right(event))
        self.master.bind("<B1-Motion>",lambda event: self.move_mouse(event))
        # child=Windows(bg='green',width=300,height=300 )


    def left (self,event):
        # self.w.leff_IA()
        if self.w.snake.dx!=1:
            self.w.snake.dx=-1
            self.w.snake.dy=0

    def right (self,event):
        # self.w.right_IA()

        if self.w.snake.dx!=-1:
            self.w.snake.dx=1
            self.w.snake.dy=0
        
    def up (self,event):
        if self.w.snake.dy!=1:
            self.w.snake.dy=-1
            self.w.snake.dx=0
        
    def down (self,event):
        if self.w.snake.dy!=-1:
            self.w.snake.dy=1
            self.w.snake.dx=0

    def move_mouse (self,event):
        pass
    
    def mousedown_left(self, event):
        self.w.snake.isdead=False
        self.start()

    def mouseup_left(self, event):
        pass
        
    def mousedown_scroll_wheel(self, event):
        self.w.snake.isdead=True
        for y in range (30):
            l="|"
            for x in range (30):
                if self.w.grid.grid[y,x]==1 : l+="O" 
                else: l+=" "
            l+="|"
            print (l)
        print('_______________________________')


    def mousedown_right(self, event):
        self.w.update()

    def start(self):
        if self.w.snake.isdead==False:
            self.w.update()
            self.w.after(SPEED,self.start)


class Windows(Toplevel):
    def __init__(self, **Arguments):
        Toplevel.__init__(self, **Arguments)
        self.title('child')
        self.geometry('+%d+%d' %(1200,50))
        self.bind("<Button-1>",lambda event: self.mousedown(event))
        self.bind("<Button-3>",lambda event: self.mousedown_right(event))

    def mousedown(self, event):
        x, y = event.x, event.y 
    
    def mousedown_right(self, event):
        x, y = event.x, event.y 


if __name__=="__main__":
    t=Window_0((30,30))
    t.mainloop()