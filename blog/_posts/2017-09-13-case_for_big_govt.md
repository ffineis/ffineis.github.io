---
layout: post
title:  "The Case for Big Brother in Big Data."
date: "2017-09-13"
excerpt: When a Data Scientists working to deploy models to entire industries or fleets should demand structure from their data. Big government, "big brother," what have you, can and should help us get to the next plateau when it comes to scaling machine learning applications beyond just one-off use cases and models.
---

### Garbage in, garbage out

Data Scientists usually wear the messiness of the data they work with as a badge of honor. And they should! It's not easy to take just any dataset and shape it into something that you can build a predictive model out of, or even just craft some sort of executive-level quantitative report. But when the task is training a model to deploy on real-time data, Data Scientists need to make sure their data was intended for more than a one-off type of analysis. We should demand structure.

Here is an impression of a dataset a Data Scientist might see in the wild -

<table>
  <caption>Made up lawnmower data. Can anyone else relate to this?</caption>
  <thead>
    <tr>
      <th>ser_no</th>
      <th>last_updated</th>
      <th>trbd</th>
      <th>price</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>U542</td>
      <td>1900/01/01 00:00:99</td>
      <td>K1242</td>
      <td>245.88</td>
      <td>342198</td>
    </tr>
    <tr>
      <td>KT294</td>
      <td>2017-09-10</td>
      <td>K1242</td>
      <td>232.98</td>
      <td>342198</td>
    </tr>
    <tr>
      <td>hello</td>
      <td>NULL</td>
      <td>test -- hold for Jim</td>
      <td>112.23</td>
      <td>342198</td>
    </tr>
    <tr>
      <td></td>
      <td>NULL</td>
      <td>yes</td>
      <td></td>
      <td>342198</td>
    </tr>
    <tr>
      <td>serial_no_234</td>
      <td>Tuesday</td>
      <td>KT234</td>
      <td>112.23</td>
      <td>342198</td>
    </tr>
  </tbody>
</table>

<br>

Maybe the people who supplied you this lawnmower data don't know what the *trbd* field means, because their DBA who quit in 2011 designed their Salesforce database. Maybe your manager wants you to use this to build a model to predict lawnmower prices based on weather patterns, but nobody can tell you if `last_updated` can safely be interpreted as when each lawnmower was sold, which is pretty crucial if you're going to try to join historical weather conditions to this price data.

My point is that unstructured, free-text data can quickly lead to the "garbage in, garbage out" phenomena.

![Dilbert](http://alison.dbsdataprojects.com/wp-content/uploads/sites/82/2016/04/cartoon-metadata.png)	

<br>

For supervised learning models, we can only expect a model to perform as well as quality of the labeled data used for training. If we don't have a clue what separates a cat from a dog sample, your model is doomed, because evaluation metrics are now meaningless. More humans can help. Humans can make heuristics to label our data. We can use Amazon Turk to get many humans to help label our data, and humans can send data to other (human) subject matter experts for label verification. But does that not defeat the entire notion of machine learning, that data needs to pass under the nose of a human before we can learn anything interesting from it?

<br>

### Big government is your friend

We should demand more from our data, and the government can help demand it. Take the electrical grid. The Federal Engery Regulatory Commission (FERC) has an immense set of guidelines for electricity producers and transmission owners when it comes to reporting as how much energy was produced and transmitted across the grid. For example, FERC requires that all electric utility companies (e.g. ones that power entire towns, cities, or regions) have to complete the [Energy Information Administration's Form 906](https://www.eia.gov/electricity/2008forms/906-923crosswalk.xls) whereby the utilities have to outline exactly how much of what types of fuels each electricity plant used over the last year. Even the units of fuel are prespecified to a selection of either "tons," "barrels," or "thousands of cubic feet." Now, Data Scientists won't need to guess the units used, or, god forbid, call the power plant and find the person who filled out the EIA 906 form to ask them about units. FERC has enforced reasonable standards in our public data, and this benefits Data Scientists by allowing us to build models that can leverage the EIA 906 form for any major electrical plant in the country, not just the handful that would have included units of input energy without the data guidelines.

<table>
  <caption>EIA 906 data for electricity utilties. A Data Scientist's dream.</caption>
  <thead>
    <tr>
      <th>PLANT NAME</th>
      <th>PLANT ID</th>
      <th>STATE</th>
      <th>PRIME MOVER TYPE</th>
      <th>ENERGY SOURCE</th>
      <th>GROSS GENERATION (Mwh)</th>
      <th>NET GENERATION (Mwh)</th>
      <th>UNITS (Tons, Barrels, or 1000s tons)</th>
      <th>STOCKS AT END OF PERIOD</th>
      <th>HEAT CONTENT PER UNIT OF FUEL (Million Btu)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sunapee</td>
      <td>AP123</td>
      <td>HIe</td>
      <td>Diesel engine</td>
      <td>Diesel fuel</td>
      <td>100.1</td>
      <td>12.5</td>
      <td>Barrels</td>
      <td>34.1</td>
      <td>6832.22</td>
    </tr>
  </tbody>
</table>

<br>

The government has solved so many other huge data headaches simply by enforcing similar data schemas. What if any publicly-traded company was able to report their annual earnings using a report template they created themselves? Apple would use one set of profit reporting guidelines while Ford used another and IBM used another one, and none of the reports would look the same. Instead, the Securities and Exchange Commission makes everyone use a 10-k form. What if there were no standards for reporting daily trading volume, high, low, and closing stock prices? We all benefit immensely from having agreed-upon and enforced data schemas.

Take just the example of a Vehicle Identificaion Number (VIN). All cars have one, and it's a standard enforced by the International Organization for Standardization - a body run by the UN, the definition of "big government." Any Data Scientist should immediately appreciate the VIN - it's an immutable, easily-interpreted primary key. If you have access to a car's VIN, you have a unique identifier for any car in addition to information about the car's characteristics that is vendor-agnostic. Without some sort of ISO convention for unique automobiles, any type of cross-vehicle make comparison would be a nightmare of ad hoc rules about what constitues an SUV, or fixing discrepancies between auto manufacturers that track the age of each car based on the year it came off of the assembly line versus manufacturers that track age based on a car's model year.

<br>

### ...the point?

I only say all of this for two reasons - (1) to defend the notion that the government does do useful things and (2) because I believe that the industries most ripe for distruption by the Internet of Things (IoT) analytics are those that are in some way overseen by some type of governmental or regulatory body. Left to their own devices, groups of competing entities building things like sensors, valves, power plants, or automobiles would most likely just create their own data and reporting conventions. This creates obstacles in letting fully human-independent machine learning take root in a way that easily scales across an industry. When "big brother" is around to nudge everyone to create, at the very least, data that vaguely resembles everyone else's data, this unlocks doors in terms of the scope of problems upon which Data Scientists can hope to deploy machine learning solutions.








