#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Exercise on Inheritance

class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
    
    def start_engine(self):
        return "The engine is starting..."
    
    def stop_engine(self):
        return "The engine is stopping..."
    
    def display_info(self):
        print(f"Car Info: {self.year} {self.make} {self.model}")

class ElectricCar(Car):
    def __init__(self, make, model, year, battery_capacity):
        super().__init__(make, model, year)
        self.battery_capacity = battery_capacity
    
    def charge_battery(self):
        return f"Charging the battery to {self.battery_capacity}"


class GasolineCar(Car):
    def __init__(self, make, model, year, fuel_tank_capacity):
        super().__init__(make, model, year)
        self.fuel_tank_capacity = fuel_tank_capacity
    
    def refuel_tank(self):
        return f"Refueling the tank to {self.fuel_tank_capacity}"


electric_car = ElectricCar("Tesla", "Model S", 2022, "100 kWh")
gas_car = GasolineCar("Ford", "Mustang", 2022, "60 liters")


electric_car.display_info()
print(electric_car.start_engine())
print(electric_car.charge_battery())
print(electric_car.stop_engine())

print()

gas_car.display_info()
print(gas_car.start_engine())
print(gas_car.refuel_tank())
print(gas_car.stop_engine())

