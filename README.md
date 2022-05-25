# Stellar-interior-numerical-model

This is my final degree project in Physics. Development of a stellar-interior numerical model built in Pyhton.

### Abstract
Development of a stellar-interior numerical model given a set of constant parametres (total mass, hydrogen and helium mass fractions) and three initial values (total radius, total luminosity and core temperature). Starting from the fundamental equations for the convective and radiative phases, they are integrated numerically following a mixed method to find the minimum total relative error in the inter-phase junction.

Python is the language selected for the construction of the model. The method is carried out by layering. First, it is integrate from the surface to the core. In a second integration the direction is reversed. Solutions are joined at the boundary between phases and the total relative error is minimized by varying the core temperature. Then, the process is repeated but now modifying total radius and total luminosity. With the help of intervals, a couple of values are found wich another interval is centered to find a more accurate solution. This step is taken to obtain the final values for total radius, total luminosity and core temperature.

With the final values, a complete model of the interior is generated for that star. Pressure, temperature, luminosity and mass are obtained for the different layers created, and ultimately as a function of radius. With these data, the model can be used to study how the different parameters can affect the different magnitudes of the star or to compare with real stars.

### Info about code

There are three files for code. The order is:
  1. **calculo_estrella_final_5**: fundamentals functions are defined in it.
  2. **calculo_errores_final_5**: find the minimum total relative error. Loops are used to work with smaller and smaller intervals and be more accurate.
  3. **calculo_final_ultimo_5**: plots and collect data to print.

Each one imports the previous file. So you can run the last file, but you have to be sure it´s all okay before start.

You don´t need any data, the program has been done to be execute by anyone who wants to. Initial values are choosen by the user who will execute the code, just... let it run! In principle it is intended for main sequence stars, but you can try to play with the code to extend for other cases.
