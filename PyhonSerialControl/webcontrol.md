# Protocol test web page

To facilitate the tests, we've developed a simple web page allowing to test the protocole. The page can be [found here](Webcontrol.py).

## Serial port initializaiton

Windows uses name like COMx where x is a number for serial ports while Linux is using names like /dev/ttyXYZ where XYZ can be S2 or USB0 or any valid tty type. So in order to make it simple to use, you can either specify the com port to use with the ```-c``` argument in the command line or setup an environement variable called COM_PORT.

Keep in mind you need root priviledges to acces serial ports in Linux.

So you can run it like ```sudo python3 Webcontrol.py -c /dev/ttyS2```. Or oy can try ```sudo chmod 666 /dev/ttyS2``` and then run the code in normal mode so ```python3 Webcontrol.py -c /dev/ttyS2```.

The code to get the serial port name thru the arguments is using ```getopt```. It first check the command line arguments which can be used either with ```-c``` either with ```--comm=```. Finally, if there is no argument, it tries to get the ```COM_PORT``` environment variable.

```python
import platform, os, sys, getopt

def getPortName(argv):
    comPort = ""
    try:
        opts, args = getopt.getopt(argv,"c:h",["com=", "help"])
    except getopt.GetoptError:
        printusage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printusage()
            sys.exit()
        elif opt in ("-c", "--com"):
            comPort = arg
        else:
            printusage()
            sys.exit(2)
    if (comPort == ""):
        try:
            comPort = os.environ["COM_PORT"]
        except KeyError: 
            printusage()
            sys.exit(2)
    return comPort
```

### Flask as web server

Flask is used as web server. A set of simple API has been created allowing to call everyone single functions from the core serial commond module.

```python
@app.route('/motor/<name>/<dir>')
def pilotmotor(name, dir):
```

They are using classical route with flask and variable names.

Because we'll be using random parameters to the request, we'll just need to clean the last part of the argument using the simple function:

```python
def cleanargs(val):
    retval = str(val).split('?')
    return retval[0]
```

## Static index file

All the script and the basic html is contained in the file ```/static/index.html```. The script is very basic. It used a classical ```XMLHttpRequest``` to run the request. A randomized parameter is added to the request to make sure the browser won't cash it.

```javascript
var xhr = new XMLHttpRequest();
var dir = 7;
var pwm = 0;
var pwminc = 15;
function btn(cmdSend) {
    xhr.open('GET', cmdSend + '?_=' + Math.random());
    xhr.send();
}
```

Then on the page simple button are coding the desired behavior.

```htlm
<input onclick="left()" type="button" value="<">
```

```javascript
function left() {
    if(dir>3) {
        dir--;
        btn("dir/"+ dir);
    }
}
```

The minimum values for the left, right are hard coded. It's just a simple and basic test page allowing to control the car when connected in wireless for example. Of course, code will need to be adjusted to create a real applicaiton using those API but work will be very quick.