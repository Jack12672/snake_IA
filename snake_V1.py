from tkinter import * 
import random
import time
import tensorflow as tf
import numpy as np
import math

SPEED=1
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
        self.border=[]
        self.fruit=0
        self.grid=Grid(dimension)
        self.snake=Snake(dimension)
        self.god=np.zeros((1,6), dtype=np.float32)
        self.snakeIA=np.zeros((1,11),dtype=np.float32)
            # input de snake
            # 0 -> obstacle devant 0/1
            # 1 -> obstacle gauche 0/1
            # 2 -> obstacle droite 0/1
            # 3 -> direction snake gauche 0/1
            # 4 -> direction snake droite 0/1
            # 5 -> direction snake haut 0/1
            # 6 -> direction snake bas 0/1
            # 7 -> direction fruit par rapport snake gauche 0/1
            # 8 -> direction fruit par rapport snake droite 0/1
            # 9 -> direction fruit par rapport snake haut 0/1
            # 10 -> direction fruit par rapport snake bas 0/1


        self.init_grid()
        self.model=0
        self.learn=True
        self.count=0
        self.predict=False
        self.save= True

        np.set_printoptions(precision=2)

        new_model=False
        self.name='SNAKE_256_256.keras'

        if new_model:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Input(shape=(11,)))
            self.model.add(tf.keras.layers.Dense(256, activation='relu'))
            self.model.add(tf.keras.layers.Dense(256, activation='relu'))
            self.model.add(tf.keras.layers.Dense(3,activation='softmax'))
            #output
            # 0 -> direction inchangée 0/1
            # 1 -> direction gauche 0/1
            # 2 -> direction droite 0/1



            self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy', 
            metrics=['accuracy']
            )

            # self.god = tf.keras.models.Sequential()
            # self.god.add(tf.keras.layers.Input(shape=(6,)))
            # self.god.add(tf.keras.layers.Dense(16, activation='relu'))
            # self.god.add(tf.keras.layers.Dense(8, activation='relu'))
            # self.god.add(tf.keras.layers.Dense(5,activation='softmax'))

        else: 
            self.model = tf.keras.models.load_model (self.name)
            # self.god = tf.keras.models.load_model ('GOD.keras')


    def distance(self,coord1,coord2)->float:
        return math.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2)  

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

    def reset_grid(self):
        self.delete("border")
        for i in range (len(self.border)-1):
            self.border.pop()
        self.border=[]
        self.delete("snake")
        for i in range (len(self.obj)-1):
            self.obj.pop()
        self.obj=[]
        self.delete("fruit")
       
        self.pack()
        self.grid=Grid(self.dimension)
        self.snake=Snake(self.dimension)
        self.init_grid()
   

    def init_grid(self):
        self.border.append(self.create_rectangle(      0, 0, 
                                                    self.dimension[0]*CASE, CASE, 
                                                    fill='orange', outline="", tags="border"))
        self.border.append(self.create_rectangle(      0, (self.dimension[1]-1)*CASE, 
                                                    self.dimension[0]*CASE, (self.dimension[1]-1)*CASE+CASE, 
                                                    fill='orange', outline="", tags="border"))

        self.border.append(self.create_rectangle(      0, 0, 
                                                    CASE, (self.dimension[1]-1)*CASE, 
                                                    fill='orange', outline="", tags="border"))
        self.border.append(self.create_rectangle(      (self.dimension[0]-1)*CASE, 0, 
                                                    (self.dimension[0]-1)*CASE+CASE, self.dimension[1]*CASE, 
                                                    fill='orange', outline="", tags="border"))

        for i in range (self.snake.length):
            x,y=self.snake.position[i]
            self.grid.grid[y,x]=1
            x,y=x*CASE,y*CASE
            if i==0:
                self.obj.append(self.create_oval(x,y,x+CASE,y+CASE,width = 1, fill="black", tags="snake"))
            else:
                self.obj.append(self.create_oval(x,y,x+CASE,y+CASE,width = 1, fill="green", tags="snake"))

        
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
            self.itemconfigure(self.obj[0],fill='black')

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
        self.fruit=self.create_oval(x0,y0,x1,y1,width = 1, fill="red", tags="fruit")

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

    def left_IA(self):
        self.snake.dx,self.snake.dy=self.get_left()

    def right_IA(self):
        self.snake.dx,self.snake.dy=self.get_right()

    def update_IA(self):
        x,y = self.snake.head
        dx,dy=self.snake.dx,self.snake.dy
        xf,yf = self.snake.fruit
        if self.count<100: self.count+=1
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
        if self.grid.grid[y_front,x_front]==1: 
            self.snakeIA[0,0]=1
        else: self.snakeIA[0,0]=0
        
        dx,dy=self.get_left()#obstacle gauche
        x_left,y_left=x+dx,y+dy
        if self.grid.grid[y_left,x_left]==1: 
            self.snakeIA[0,1]=1
        else: self.snakeIA[0,1]=0

        dx,dy=self.get_right() #obstacle droite
        x_rignt,y_right=x+dx,y+dy
        if self.grid.grid[y_right,x_rignt]==1:
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
 

 
        if self.learn:
            ep=1
            # score=np.zeros((1,3),dtype=np.float32)
            x,y = self.snake.head
            xf,yf = self.snake.fruit
            
                # tout droit
            dx,dy=self.snake.dx,self.snake.dy
            x_front=x+dx
            y_front=y+dy
                # à gauche
            dx,dy=self.get_left()
            x_left=x+dx
            y_left=y+dy
                # à droite
            dx,dy=self.get_right()
            x_right=x+dx
            y_right=y+dy

            # test mange fruit
            if x_front==xf and y_front==yf:
                score=np.array([[1, 0, 0]], dtype=np.float32)          
            elif x_left==xf and y_left==yf:
                score=np.array([[0, 1, 0]], dtype=np.float32)
            elif x_right==xf and y_right==yf:
                score=np.array([[0, 0, 1]], dtype=np.float32)
            else: #test obstacles
                dist_front=self.distance ((x_front,y_front),(xf,yf))
                dist_left=self.distance ((x_left,y_left),(xf,yf))
                dist_right=self.distance ((x_right,y_right),(xf,yf))
                possibles=[]
                possible_front=1
                possible_left=1
                possible_right=1
                if self.grid.grid[y_front,x_front]==1: possible_front=10000
                if self.grid.grid[y_left,x_left]==1: possible_left=10000
                if self.grid.grid[y_right,x_right]==1: possible_right=10000

                possibles.append((dist_front*possible_front,'front'))
                possibles.append((dist_left*possible_left,'left'))
                possibles.append((dist_right*possible_right,'right'))

                possibles=sorted(possibles,key=lambda dist: dist[0])
                
                if possibles[0][1]=='front':
                    score=np.array([[1, 0, 0]], dtype=np.float32)          
                elif possibles[0][1]=='left':
                    score=np.array([[0, 1, 0]], dtype=np.float32)
                elif possibles[0][1]=='right':
                    score=np.array([[0, 0, 1]], dtype=np.float32)

            history = self.model.fit(self.snakeIA, score,epochs=ep, verbose=0)


        # print(self.snakeIA)
        predic=self.model.predict(self.snakeIA,verbose=0)
        pos=np.argmax(predic)
        if pos==1: self.left_IA() 
        if pos==2: self.right_IA()


    def update(self):
            self.update_IA()
            x,y=self.snake.position[0][0]+self.snake.dx,self.snake.position[0][1]+self.snake.dy
            self.snake.head=(x,y)
            growth=False
            
            if self.check_obstacle():
                print('end')
                if self.save:
                    self.model.save (self.name)
                self.reset_grid()

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
            self.delete("fruit")
            l=self.snake.length
            x,y=self.snake.position[l-1][0],self.snake.position[l-1][1]
            self.snake.position.append((x,y))
            x,y=x*CASE,y*CASE
            self.obj.append(self.create_oval(x,y,x+CASE,y+CASE,width = 1, fill="green", tags="snake"))
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
        self.master.bind("<KeyPress-space>",lambda event: self.space(event))
        self.master.bind("<KeyPress-a>",lambda event: self.key_a(event))
        self.master.bind("<KeyPress-s>",lambda event: self.key_s(event))

        self.master.bind("<Button-1>",lambda event: self.mousedown_left(event))
        self.master.bind("<Button-2>",lambda event: self.mousedown_scroll_wheel(event))
        self.master.bind("<Button-3>",lambda event: self.mousedown_right(event))
        self.master.bind("<B1-Motion>",lambda event: self.move_mouse(event))
        # child=Windows(bg='green',width=300,height=300 )


    def left (self,event):
        # self.w.left_IA()
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

    def space (self,event):
        x,y=self.w.snake.head
        dx,dy=self.w.get_left()
        if self.w.grid.grid[dy+y,dx+x]==0:
            self.w.left_IA()
        else: self.w.right_IA()
        self.w.snake.isdead=False
        self.start()

    def key_a (self,event):
        self.w.test_aff()

    def key_s (self,event):
        self.w.test_sup()

    def move_mouse (self,event):
        pass
    
    def mousedown_left(self, event):
        self.w.snake.isdead=False
        self.start()

    def mouseup_left(self, event):
        pass
        
    def mousedown_scroll_wheel(self, event):
        self.w.snake.isdead=True
        for y in range (self.w.dimension[1]):
            l="|"
            for x in range (self.w.dimension[0]):
                if self.w.grid.grid[y,x]==1 : l+="O" 
                else: l+=" "
            l+="|"
            print (l)
        print('_______________________________')


    def mousedown_right(self, event):
        self.w.update()

    def start(self):
        # if self.w.snake.isdead==False:
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
    t=Window_0((15,10))
    t.mainloop()