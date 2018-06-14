from subprocess import call
import numpy as np
def main():
    print("you suck")
    for i in range(250):
        n = i * 100
        with open('./test.ifd') as f:
            contents = f.read().replace('bbtemp 4177', 'bbtemp ' + str(n)).replace('bbtemp = 5000', 'bbtemp = ' + str(n))
        with open('./ifdsTemp/render_fire_{}.ifd'.format(n), "w+") as f:
            f.write(contents)
        print("No You Suck! Epoch {}".format(i))    
        f = open('/dev/null', 'w')
        call(["mantra", './ifdsTemp/render_fire_{}.ifd'.format(n), "fireTemp/gen_{}.jpg".format(n)],stderr=f)
        f.close()


if __name__ == '__main__':
    main()