import math
from django.shortcuts import render
from .forms import AdditionForm,NumericalIntegrationForm,InterpolationForm,UnequalForm# Import your form
from decimal import Decimal
import numpy as np


def linear_regression(request):
    r, m, c, a = None, None, None, None  # Initialize variables with default values
    str = None
    result_displayed = False

    if request.method == 'POST':
        form = AdditionForm(request.POST)
        if form.is_valid():
            number1 = form.cleaned_data['number1']
            number2 = form.cleaned_data['number2']
            x = []
            y = []

            nx = number1.split(",")
            ny = number2.split(",")
            for i in range(len(nx)):
                x.append(float(nx[i]))
                y.append(float(ny[i]))
            r, m, c = apprun(x, y)
            str = correl(r)
            
            if "yonx" in request.POST.get("action"):
                a = yonx(r, m, c)
            elif "xony" in request.POST.get("action"):
                a = xony(r, m, c)
            
            result_displayed = True  # Set this flag to indicate that results should be displayed

    else:
        form = AdditionForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'r': r,
        'm': m,
        'c': c,
        'str': str,
        
    }

    return render(request, 'linear_regression.html', context)

def apprun(x,y):
    n = len(x)
    Exy = 0
    Ex = 0
    Ey =0
    xm=sum(x)/n
    ym=sum(y)/n
    for i in range(n):
        Exy = Exy + (x[i]-xm)*(y[i]-ym)
        Ex = Ex + pow(x[i]-xm,2)
        Ey = Ey + pow(y[i]-ym,2)
    r = Exy/(math.sqrt(Ex*Ey))
    byx = r * math.sqrt(Ey/Ex)
    c = ym - xm * byx
    return r,byx,c

def correl(r):
    if r == 0:
        return "No Correlation"
    elif r>0:
        # st.subheader(f'r = {round(r,3)}')
        return "Correlation is Positive"
    else:
        # st.subheader(f'r = {round(r,3)}')
        return "Correlation is negative"
def yonx(r,m,c):
    if r!=0:
        newstr = f"y = {round(m,3)}x "
        if c>0:
            newstr = newstr + f" + {round(c,3)}"
        elif c<0:
            newstr = newstr + f" {round(c,3)} "
   
    return (f"{newstr} ")

def xony(r,m,c):
    if r!=0:
        newstr = f"x = {round(m,3)}y "
        if c>0:
            newstr = newstr + f" + {round(c,3)}"
        elif c<0:
            newstr = newstr + f" {round(c,3)} "
    return (f"{newstr} ")




def numerical_integration(request):
    step, y_values, result, error_message = None, None, None, None
    result_displayed = False

    if request.method == 'POST':
        form = NumericalIntegrationForm(request.POST)
        
        if form.is_valid():
            y = []
            step = form.cleaned_data['step']
            y_values = form.cleaned_data['yvalues']
            ny = y_values.split(",")
            for i in range(len(ny)):
               y.append(float(ny[i]))
            n = len(y)
            
            if "trapezoidal" in request.POST.get("action"):
                result = trapezoidal_rule(step, y)
                error_message = (f"The value of the Integral is {result}")
            elif "simpson 1/3" in request.POST.get("action"):
                if (n - 1) % 2 == 0:
                    result = simpsons_one_third_rule(step, y)
                    error_message = (f"The value of the Integral is {result}")
                else:
                    error_message = "Cannot apply Simpson 1/3rd rule on the given data."
            elif "simpson 3/8" in request.POST.get("action"):
                if (n - 1) % 3 == 0:
                    result = simpsons_three_eighth_rule(step, y)
                    error_message = (f"The value of the Integral is {result}")
                else:
                    error_message = "Cannot apply Simpson 3/8th rule on the given data."
            result_displayed = True

    else:
        form = NumericalIntegrationForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str': error_message,
    }
    return render(request, 'numerical_integration.html', context)

def trapezoidal_rule(step, y):
    part1 = y[0] + y[-1]
    part2 = sum(y[1:-1])
    area = step * (part1 + 2 * part2) / 2
    return round(area, 6)

def simpsons_one_third_rule(h, y):
    n=len(y)
    part1 = y[0] +y[n-1]
    part2 = 0
    part3 = 0
    for i in range(1,n-1):
        if i%2==0:
            part2 = part2 + y[i]
        else:
            part3 = part3 + y[i]
    area = h*(part1 + 2*part2 + 4*part3)/3
    return round(area,6)

def simpsons_three_eighth_rule(step, y):
    part1 = y[0] + y[-1]
    part2 = sum(y[1:-1:3]) * 3
    part3 = sum(y[2:-2:3]) * 3
    part4 = sum(y[3:-3:3])
    area = 3 * step * (part1 + part2 + part3 + part4) / 8
    return round(area, 6)


def demo(request):
    return render(request,"sidebar.html")



def interpolation_polynomial(request):
      # Initialize variables with default values
    str = None
    result_displayed = False

    if request.method == 'POST':
        form = InterpolationForm(request.POST)
        if form.is_valid():
            xvalue = form.cleaned_data['xvalue']
            yvalue = form.cleaned_data['yvalue']
            x = []
            y = []
            flag=0
            con=-1
            nx = xvalue.split(",")
            ny = yvalue.split(",")    

            for i in range(len(nx)):
                x.append(float(nx[i]))
                y.append(float(ny[i]))
            for i in range(len(x)):
                if x[i]==0:
                    con=y[i]
                    y.pop(i)
                    x.pop(i)
                    flag=1
                    break
            str=apprun3(len(x),x,y,flag,con)           
            result_displayed = True  # Set this flag to indicate that results should be displayed
    else:
        form = InterpolationForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str': str,
    }

    return render(request, 'interpolation_polynomial.html', context)

def apprun3(n,xn,yn,flag,con):
    a = np.zeros((n, n + 1))
    x = np.zeros(n)
    z = np.zeros(n)
    if flag==0:
        for i in range(n):
            for j in range(n):
                a[i][j] = pow(xn[i], j)

        for i in range(n):
            a[i][n] = yn[i]
    else:
        for i in range(n):
            for j in range(n):
                a[i][j] = pow(xn[i],j+1)

        for i in range(n):
            a[i][n] = yn[i]-con

    for i in range(n):
        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]
            for k in range(n + 1):
                a[j][k] = a[j][k] - ratio * a[i][k]

    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = a[i][n]

        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]

        x[i] = x[i] / a[i][i]

    for i in range(n):
        z[i] = round(x[i], 3)
   
    newstr = "y= "
    if flag==0:
        for i in range(n, 1, -1):
            if (z[i-1]>0 and newstr == "y= ") or z[i-1]<0:
                newstr = newstr + f" {z[i - 1]}x^{i - 1} "
            elif z[i-1]>0:
                newstr = newstr + f" + {z[i -1]}x^{i - 1}"
        if z[0]>0 and newstr != "y= ":
            newstr = newstr + f" + {z[0]} "
        elif z[0]<0 or newstr == "y= ":
            newstr = newstr + f" {z[0]}"
    else:
        for i in range(n,0,-1):
            if (z[i-1]!=0 and newstr == "y= ") or z[i-1]<0:
                newstr = newstr + f" {z[i-1]}x^{i}  "
            elif z[i-1]>0:
                newstr = newstr + f" + {z[i-1]}x^{i} "
        if con>0 and newstr == "y= ":
            newstr = newstr + f" + {con} "
        elif con<0:
            newstr = newstr + f" {con} "
    return (f" {newstr} ")



def interpolation_unequal(request):
    result_displayed = False
    str=None

    if request.method == 'POST':
        form = UnequalForm(request.POST)
        
        if form.is_valid():
            x = []
            y = []
            flag=0
            sx = form.cleaned_data['xvalues']
            sy = form.cleaned_data['yvalues']
            xp = form.cleaned_data['poi']
            nx = sx.split(",")
            ny = sy.split(",")

            for i in range(len(nx)):
                x.append(float(nx[i]))
                y.append(float(ny[i]))
            if "Lagrange Interpolation" in request.POST.get("action"):
                str=apprun4(len(x),x,y,xp)
            elif "Divided Difference" in request.POST.get("action"):
                str=divided(len(x),x,y,xp)                  
            result_displayed = True

    else:
        form = UnequalForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str':str  
    }
    return render(request, 'interpolation_unequal.html', context)

def apprun4(n,x,y,xp):
    num=[]
    den=[]
    s=0
    for i in range(n):
        a = 1
        b = 1
        for j in range(n):
            if i != j:
                a = a * (xp - x[j])
                b = b * (x[i] - x[j])
        num.append(a)
        den.append(b)
    for i in range(n):
        s=s+y[i]*num[i]/den[i]
    return (f"The value of y at x = {xp} is equal to {round(s,3)}")
def divided(n,x,y,xp):
    s = 0
    c = []
    c.append(1)
    arr = np.zeros((n, n))
    for i in range(n):
        arr[0][i] = y[i]
    for i in range(1, n):
        for j in range(n - i):
            arr[i][j] = (arr[i - 1][j + 1] - arr[i - 1][j])/(x[j+i] - x[j])
    for i in range(n):
        a=1
        for j in range(i+1):
            a=a*(xp-x[j])
        c.append(a)
    for i in range(n):
        s=s+arr[i][0]*c[i]
    return (f"The value of y at x = {xp} is equal to {round(s,3)}")



def newtongregory(request):
    result_displayed = False
    str=None

    if request.method == 'POST':
        form = UnequalForm(request.POST)
        
        if form.is_valid():
            x = []
            y = []
            flag=0
            sx = form.cleaned_data['xvalues']
            sy = form.cleaned_data['yvalues']
            x0 = form.cleaned_data['poi']
            nx = sx.split(",")
            ny = sy.split(",")

            for i in range(len(nx)):
                x.append(float(nx[i]))
                y.append(float(ny[i]))
            if "Forward Interpolation" in request.POST.get("action"):
                str=app1(len(x),x,y,x0)
            elif "Backward Interpolation" in request.POST.get("action"):
                str=app2(len(x),x,y,x0)                  
            result_displayed = True

    else:
        form = UnequalForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str':str      
    }
    return render(request, 'newtongregory.html', context)

def app1(n,x,y,x0):
    p=(x0-x[0])/(x[1]-x[0])
    s=0
    arr=np.zeros((n,n))
    for i in range(n):
        arr[0][i]=y[i]
    for i in range(1,n):
        for j in range(n-i):
            arr[i][j]=arr[i-1][j+1]-arr[i-1][j]
    coef=value(p,n)
    for i in range(n):
        s=s+arr[i][0]*coef[i]
    newsum=round(s,3)
    return (f"The value of y at x = {x0} is equal to {newsum}")
def value(p,n):
    c=[]
    c.append(1)
    for i in range(1,n):
        c.append(chain(p,i)/fact(i))
    return c
def chain(p,n):
    if n>1:
        return (p-(n-1))*chain(p,n-1)
    else:
        return p
def fact(n):
    if n>0:
        return n*fact(n-1)
    else:
        return 1
def app2(n,x,y,x0):
    p = (x0 - x[n-1]) / (x[1] - x[0])
    s = 0
    arr = np.zeros((n, n))
    for i in range(n):
        arr[0][i] = y[i]
    for i in range(1,n):
        for j in range(i,n):
            arr[i][j] = arr[i - 1][j] - arr[i - 1][j - 1]
    coef = value2(p, n)
    for i in range(n):
        s = s + arr[i][n-1] * coef[i]
    newsum = round(s, 3)
    return (f"The value of y at x = {x0} is equal to {newsum}")
def value2(p,n):
    c = []
    c.append(1)
    for i in range(1, n):
        c.append(chain2(p, i) / fact(i))
    return c
def chain2(p,n):
    if n>1:
        return (p+(n-1))*chain2(p,n-1)
    else:
        return p



def numerical_differentiation(request):
    result_displayed = False
    str=None

    if request.method == 'POST':
        form = UnequalForm(request.POST)
        
        if form.is_valid():
            x = []
            y = []
            flag=0
            sx = form.cleaned_data['xvalues']
            sy = form.cleaned_data['yvalues']
            x0 = form.cleaned_data['poi']
            nx = sx.split(",")
            ny = sy.split(",")

            for i in range(len(nx)):
                x.append(float(nx[i]))
                y.append(float(ny[i]))
            if "Calculate" in request.POST.get("action"):
                str=apprun5(len(x),x,y,x0)
            result_displayed = True
           
    else:
        form = UnequalForm()

    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str':str
            
    }
    return render(request, 'numerical_differentiation.html', context)


def apprun5(n,x,y,x0):
    p=(x0-x[0])/(x[1]-x[0])
    h=x[1]-x[0]
    s=0
    arr=np.zeros((n,n))
    fy=[]
    for i in range(n):
        arr[0][i]=y[i]
    for i in range(1,n):
        for j in range(n-i):
            arr[i][j]=arr[i-1][j+1]-arr[i-1][j]
    for i in range(1,n):
        fy.append(arr[i][0])
    ci=val(p,n)
    for i in range(n-1):
        s=s+fy[i]*ci[i]/fact(i+1)
    newsum=round(s/h,3)
    return (f"The derivative of y at x = {x0} is equal to {newsum}")


def val(p,n):
    c=[]
    c.append(1)
    for i in range(2, n):
        new = 0
        for j in range(i):
            a = 1
            for k in range(i):
                if k != j:
                    a = a * (p - k)
            new = new + a
        c.append(new)
    return c
def fact(n):
    if n>0:
        return n*fact(n-1)
    else:
        return 1


def curve_fitting(request):
      # Initialize variables with default values
    str = None
    result_displayed = False

    if request.method == 'POST':
        form = InterpolationForm(request.POST)
        if form.is_valid():
            xvalue = form.cleaned_data['xvalue']
            yvalue = form.cleaned_data['yvalue']
            x = []
            y = []
            nx = xvalue.split(",")
            ny = yvalue.split(",")  
            n = len(nx) 

            if "y=ax+b" in request.POST.get("action"):
                assign(nx,ny,x,y)
                a,b=linear(x,y,n)
                str=(f"$$ y=({a})x+({b}) $$")
            elif "y=ax^2+bx+c" in request.POST.get("action"):
                assign(nx,ny,x,y)
                a,b,c=quadratic(x,y,n)
                str=(f"$$ y=({a})x^2+({b})x+({c}) $$")
            elif "y=ab^x" in request.POST.get("action"):
                assign(nx,ny,x,y)
                lny = []
                for i in range(n):
                    lny.append(math.log(y[i],2))
                lnb,lna = linear(x,lny,n)
                a = round(pow(2,lna),3)
                b = round(pow(2,lnb),3)              
                str=(f"$$ y=({a})({b})^x $$")
            elif "y=ax^b" in request.POST.get("action"):
                assign(nx,ny,x,y)
                lnx = []
                lny = []
                for i in range(n):
                    lnx.append(math.log(x[i],2))
                    lny.append(math.log(x[i],2))
                b,lna=linear(lnx,lny,n)
                a = round(pow(2,lna))
                str=(f"$$ y=({a})x^({b}) $$")
            elif "y=1/(ax+b)" in request.POST.get("action"):
                assign(nx,ny,x,y)
                recy = []
                for i in range(n):
                    recy.append(1/y[i])
                a,b=linear(x,recy,n)
                str=(f"$$ y=1/[({a})x+({b})] $$")
            result_displayed = True
    else:
        form = InterpolationForm()
    context = {
        'form': form,
        'result_displayed': result_displayed,
        'str': str,
    }

    return render(request, 'curve_fitting.html', context)


def linear(x,y,n):

    Exy=0
    Ex2=0
    Ey = sum(y)
    Ex = sum(x)
    for i in range(n):
        Exy = Exy + x[i]*y[i]
        Ex2 = Ex2 + pow(x[i],2)
    det = (n * Ex2 - Ex * Ex)
    detb = (Ex2 * Ey - Ex * Exy)
    deta = (Exy * n - Ex * Ey)
    a = deta/det
    b = detb/det
    return round(a,3),round(b,3)

def quadratic(x,y,n):
    Ey = sum(y)
    Ex = sum(x)
    Exy = 0
    Ex2y = 0
    Ex2 = 0
    Ex3 = 0
    Ex4 = 0
    for i in range(n):
        Exy = Exy + x[i]*y[i]
        Ex2y = Ex2y + pow(x[i],2)*y[i]
        Ex2 = Ex2 + pow(x[i],2)
        Ex3 = Ex3 + pow(x[i],3)
        Ex4 = Ex4 + pow(x[i],4)
    det = Ex2*(Ex2*Ex2 - Ex*Ex3) - Ex*(Ex2*Ex3 - Ex*Ex4) + n*(Ex3*Ex3 - Ex2*Ex4)
    deta = Ey*(Ex2*Ex2 - Ex*Ex3) - Ex*(Ex2*Exy - Ex*Ex2y) + n*(Ex3*Exy - Ex2*Ex2y)
    detb = Ex2*(Ex2*Exy - Ex*Ex2y) - Ey*(Ex2*Ex3 - Ex*Ex4) + n*(Ex3*Ex2y - Ex4*Exy)
    detc = Ex2*(Ex2*Ex2y - Ex3*Exy) - Ex*(Ex3*Ex2y - Ex4*Exy) + Ey*(Ex3*Ex3 - Ex4*Ex2)
    a = deta/det
    b = detb/det
    c = detc/det
    return round(a,3),round(b,3),round(c,3)
def assign(nx,ny,x,y):
    for i in range(len(nx)):
        x.append(float(nx[i]))
        y.append(float(ny[i]))
