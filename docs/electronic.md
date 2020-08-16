# Car electronic

The electronic is an important part of the car. There are few important elements which you need to take care of:

- choice or boards
- electrical alimentation for the various boards
- motor power drivers and electrical alimentation
- direction of the car
- on/off
- connections
- batteries
- webcam
- cooling down everything

And of course, all this needs to be at a reasonable cost!

![the car](/docs_image/thecar.jpg)

As you can see the car is mounted with screws and acrylic glass (can be done with Plexiglas as well). Advantage of acrylic is that it's more robust than the Plexiglas. When cutting and making the holes the acrylic, keep the plastic protection so you won't damage your work. Note that it takes about 1 day to cut, make the holes and assemble all what is needed to make the car nice. Make sure when you’ll assemble everything, you'll screw strongly everything, the vibration of the car will clearly unscrew them if not done properly. This can damage the electronic or have component going out of the car while driving!

Note that the first prototype was done with hard paper and glue gun to make is fast on prototyping.

![the old car](/docs_image/oldcar.jpg)

## Choice of boards

This has been a very hard choice. There are multiple options and we've tried to summarize all in this table.

| Board type | Advantages | Inconvenient | Other elements |
| --- | --- | --- | --- |
| Raspberry | - very popular platform | - slow processing, so no training on the board | - armhf architecture|
|| - lots of available code, libraries | - consumption quite high ||
||| - price quality low 50€ naked ||
||| - lack of support for OpenCV, Tensorflow and other advanced libriray | - but all available for recompilation |
| Arduino, ESP8266 | - perfect for low level hardware piloting | - no camera support | - AtMega |
|| - lots of available code, libraries | - only C/C++ development | - lots of form factors, from micro to mega |
|| - great connectors, native PWM, serial... | - can't load any Tensorflow model||
|| - extremely low cost|||
| Embedded PC (ex Z83) | - native Windows/Linux x86/64 support | - cost starting at 100€ | - x86/64 |
|| - supports natively webcams, Tensorflow | - usually limited number of IO, no serial ||
||| - important electrical consumption, close to 1A ||
| Embedded ARM based board (ex RockPro64) | - power/cost is just great! 80€ naked | - not yet popular platform, limited support, relatively active wiki | - arm64v8 (aarch64) |
|| - native GPIO, serial, webcam, USB3, PCI Port, eMMC and SD card | - lack of support for OpenCV, Tensorflow and other advanced library | - but all available for recompilation |
|| - very low power consumption, less than 500mA with all cores running! || - lots of Android and Linux OS available |
| Using a Phone or equivalent | - quite cheap for low end ones | - limited customization support, would need to use Android with old Linux versions | - More Android than iPhone, Java for app development |
|| - great at managing consumption with its own battery | - limited serial support, only Bluetooth ||
||| - too limited power to run the models ||
||| - quite costly for high end phones ||

The Raspberry solution can be working and has been quickly tested but the processing is limited. So as soon as you want to do a bit more than using an existing model, it is too limited. IO on the SD are limited and won't allow to save all images took from the webcam while the car will be driving for example. Full classes to pilot the car has been implemented for this testing purpose and can be found [here](RaspberryPiControl).

So based on those various choices, the fact we needed a board with a fast webcam transfer, fast processing for the images, support for basic IO, we've decided to go for the [RockPro64 from Pine64](http://www.pine64.org). Cost for the board is 80$ so when adding the shipment, you arrive at 80€.

![rockpro64](/docs_image/ROCKPro64_slide.jpg)

We've decided as well to use a simple Arduino for the low level hardware piloting taking advantages of very low cost (3€) for a mini Arduino, easy development, easy flash, easy to send orders thru serial port. Making a separation between the main RockPro64 board, the motors and the rest of the low level electronic.

![arduino](/docs_image/arduino.jpg)

## Electrical consumption

### RockPro64

The good news is that the RockPro64 consumption is less than 500 m4 even with the webcam, a wifi dongle. So alimentation can be directly using the main battery if it is around 12V. Inconvenient when shared with a common battery is that motors are generating high frequencies which are not filtered by the board (looking at the simple design). so we decided to go for a specific step up converter up to 80W. So with 12V, it is a very comfortable 6A output. Much more than needed.

This converter can be found easily on sites like Banggood for less than 3 euros. For example, it looks like:

![image](/docs_image/converterup.png)

You can easily regulate the output with a small screw driver and a voltmeter. Using this converter allow to make sure the board will be isolated in terms of electricity from the motors as well having a secured regulated tension all the time with 80W max so 6A at 12V as needed.

Note: the tension regulator to convert 7.2V to 12V uses a small screw. With the car vibration the screw slowly unscrew and increased the tension to more than 30V which burned the 2 converters on the board. This fully killed the board and we then lost one of this fabulous board.To avoid similar issue, you need to **physically glue the screw on the tension regulator**. Because we were short in time and because we were so sad to loose the board that w've decided to temporary use a Raspberry Pi to pilot the car.

### Arduino

Arduino mini consumption is very minimum, around 40mA for what we are doing using PWM for the motor, the servo motor, few GPIO and serial port. So, we can use the 3.3V alimentation from the RockPro64 board (pin 1). The RockPro can deliver it without any problem.

Choice of 3.3V is dictated by the RockPro which uses 3.3V pin out.

## Main motor

Originally, we wanted to use a classical 2A h-bridge to pilot the motor and use directly the battery alimentation for that. The Arduino will pilot the PWM part and the pins.

After few tests in real life, we realize that it is not powerful enough to move the car! We thought maybe the tension was not 7.2V but much higher. We were quite surprised and ran tests on the old electronic. The old electronic just uses relays without any PWM or equivalent and tension was the 7.2V directly.

So, we measured the intensity that motors are asking and it is about 4A! Yes, **the main motor requires 4A**. While our h-bridge delivers a maximum of 2A (measurement shows 2.4A), the solution was to use both h-bridges present in parallel to increase the max intensity delivered to the motors. And **it works :-)**.

So all up cost to pilot the motors is the double hbridge already fully created which costs 4 € on Banggood.

![hbridge](/docs_image/hbridge.png)

Now, during the race and after some usage, this h-bridge was too limited and the current was higher than the 4A over heating the hbridge which then just stopped working. So we choose another h-bridge with a max 43A current which was more than largely sufficient :-). The new module is based on BTS7960 with 1 very large heat dissipator.

![hbridge](/docs_image/BTS7960-Modules-2.jpg)

Documentation of the board is far from perfect so it took a bit of time to understand how best to use it. See the full schematic to understand the usage. The most disturbing part is the naming on the board with PWM entries while they are the selection for the hbridge and the EN part is actually the PWM! So all up, if you follow the documentation, you'll have to use 2 PWM and always select the hbridge while the reality is different, you have to select the hbridge and then use the PWM for the entry.

## Direction of the car

Originally the car had a standard 5V motor and a potentiometer made with resistor to understand the direction of the car. This system is not precise enough and have a lot of inconvenient.

so we've decided to replace it with a servo motor. That said, the servo motor needed to commute fast as well as being powerful enough to turn the engine.

Because of the size of the servo vs the original direction, the speed and power, we've decided to reuse a servo we already had. It's a standard Modelcraft MC-410. Cost approx 6.50 €.

The servo needs 5V and can consume up to 500 mA at peak (and can be even more). Originally, in the first test, it was alimented from the h-bridge. But the solution was not powerful enough and the alimentation felt during high demand. So it has its own 5V alimentation (shared with a fan) with a 7805 regulator associated with a large few 100s µF condensator to limit the peaks. The solution is very reliable and works now perfectly. Cost is about 1 €.

Even if the Arduino pin out is 3.3V (when alimentation is 3.3V), the low signal limit on the servo motor is less than 2.5V so 3.3V is seen as a high signal which is a good new as no tension converter is needed in this case.

On top of the electronic, a bit of mechanic has been needed to reuse the existing direction and as many elements as possible.

![servo](/docs_image/servomc410.png)

## On/Off

Sometimes tiny details can make a difference like a simple on/off button. Why? Well, the main battery can be removed stopping everything to work. But we've realized as well that the embedded serial ports we are using was used at boot time by the Linux kernel to send the debug information wand we didn't find a way to get rid of it.

So during the boot, the Arduino is stopped with a simple on/off switch plugged on the 3.3V (or 5V in the Rapsberry Pi version as pins are 5V tolerant). Once the board is booted and the program launched, it is safe to switch the Arduino on. We've been reusing fully the embedded switch and just changed the wires.

## Connections

As you can read, there are now quite a lot of element to connect to each others. For the battery, we're using a standard modelist connector, the one which was existing in the car we've reused.

To connect the motors to the hbridge, we're using standard electrical cables, same for the battery to the hbridge.

From hbridge to the board and to the 5V converter, we're using standard cables from network/phone cables. Fully reused from broken ones. It is largely enough regarding the intensity used by the various elements. The physical connection uses special plugs so it limits errors when plugging the alimentations. Note that there is no diode protection on the electronic so far. Something to add maybe.

The board is connected to the Arduino thru a cable tablecloth. Only 4 cables are used (3.3V, ground, RX, TX) and ironed directly on the electronic board. Another option would have been to use a symmetrical cable tablecloth and more connectors on the electronic board, just being lazy to iron more connectors :-)

## Batteries

Batteries are an important part of the car, they will have to deliver enough energy to the various boards, cards, and of course the motors. We've been for classical modelist batteries. A rechargeable 4500 mA 7.2V has been chosen. Those are high ends batteries. Cost is quite high, count 26€.

The choice has been made easy as we've been reusing an old car for the race, removing the 'old' electronic, keeping some of the cables. So we already had a battery available as well as the charger.

As those batteries are used in the remote controlled cars, we know that they can delivered the desired power and sustain for more time than needed.

Note: after the accident which burned the RockPro 64, we've decided to use a Raspberry Pi which will be connected to a power battery pack. It makes it simple and efficient for this very of the car.

## Webcam

After reusing an old USB webcam. We've decided to go for a a wide angle camera. The old webcam did the trick but the quality was not good enough and the lengths were hard to setup. So we bought a wide angle camera. The webcam specification are 1080P H.264 CMOS 30fps 180 degrees Fisheye, low light. It has a USB connector and is connected thru USB to the board. It worked with the RockPro64 as well as with the Raspberry Pi. Cost of good quality module for webcams is quite high but necessary for good quality images.

There are other options that we're currently looking at for the webcam. Like a normal webcam which will be moved up or down depending on the speed of the car to see closer or further and better anticipate the turns. The wide angle camera has the inconvenient to disform quite a lot the images which makes is a bit harder to tag and interpret for AI. See the [AI section](/ai.md) for more information.

## Cooling down everything

There are 2 elements which heat quite a lot when used: the main rockpro64 processor and the double hbridge when the car is driving as the max 4A are used.

At first, we did cool the rockpro with a small radiator and a fan but moved to a fan less solution with a full large radiator. This is part of the additional elements we bought for the rockpro board.

The second element which heat quite a lot is the double hbridge. It already has its own radiator. We're using as well a small fan to increase the cooling.

## Costs

Here is the list of components we bought and those reused.

| Element | Quantity | Cost | Total cost |
| --- | --- | --- | --- |
| Battery 4500 mA 7.2V NiMh | 1 | 26 € | 26 € |
| RockPro64 (for first version before it burned) | 1 | 80 € | 80 € |
| Additional board components, eMMC, Webcam, radiators | x | 80 € | 80 € |
| 32G SD card | 1 | 12 € | 12 € |
| Arduino mini | 1 | 3 € | 3 € |
| Double h-bridge 2x2A | 1 | 5 € | 5 € |
| 5V alimentation | 1 | 1 € | 1 € |
| 7.2 to 12V converter | 1 | 3 € | 3 € |
| Wide angle Webcam | 1 | 63 € | 63 € |
| Raspberry Pi (for second version) | 1 | 50 € | 50 € |
| Acrylic glass (didn't find a small one) | 1 | 30 € | 30 € |
| Various screw all diameters | | 10 € | 10 € |

Reused elements

| Element | Quantity | Cost | Total saving |
| --- | --- | --- | --- |
| Old Christmas present radio guided car | 1 | priceless :-) | 70 € |
| Battery 3000 mA 7.2V NiMh | 1 | 15 € | 15 € |
| Charger for NiMh batteries | 1 | 40 € | 40 € |
| Servo motor Motorcraft MC410 | 1 | 6.50 € | 6.50 € |
| Cable tablecloth | 1 | 1 € | 1 € |
| Cables, various components, connectors | x | 5 € | 5 € |
| Reused old web cam and lenses | x | 10 € | 10 € |

So, all up, we bought for 363 € and reused for about 147.50 €, so a total cost of 510,50 €. And this cost doesn't count some extra hardware, few more electronic components. All will be fully reused anyway :-) And part of the increased cost is due to the fact we've burned one board, the RockPro64 :'(

## Electronic schema

The full electronic schema looks like that for the first version:

![schema](/docs_image/schema.png)

Few comments:

- To couple both output of the double h-bridge, make sure you first measure the tension at each output before linking them. If you don't do it, and the internal schema is inverted, you risk to fully burn the hbridge.
- To use correctly the PWM for the hbridge, please make sure you remove the 2 jumpers which are attached to it.
- Use the internal 5V power converter for the hbridge to power it internally, it is enough for it but don't use it for a servo motor or any other power element, it is not enough. It can be enough for an Arduino mini for example.
- You may have to invert the motor input depending on the way it goes so forward will be forward and backward is backward as well.

The full schematic is like that for the second version running on Raspberry Pi:

![schema](/docs_image/schema2.png)

Few comments:

- The RPI is running on a battery pack, an 7.2V->5V transformer can be used instead. Please make sure you'll glue the screw of the resistor used to have the 5V output. Vibrations of the car will change it for sure.
- We've been using a 43A hbridge

Picture of the main electronic board where the Arduino, the 5V power alimentation and the connectors are located:

![main board](/docs_image/board1.jpg)

![main board](/docs_image/board2.jpg)
