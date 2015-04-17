Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  1 
Loop Back Test Status    Integer*2  N/A  0 to 3  1  0=Pass, 1=Fail, 2=Timeout, 3=Not Tested (1)  2 
T1 Output Frames  The number of octets received on interface, including frame octets  Integer*4  octet  0 to 232-1  1  N/A  3 - 4 
T1 Input Frames  The number of octets sent on interface, including frame octets  Integer*4  octet   0 to 232-1  1  N/A  5 - 6 
Router Memory Used  Bytes currently in use by applications on managed device  Integer*4  byte   0 to 232-1  1  N/A  7 - 8 
Router Memory Free  Bytes currently free on managed device  Integer*4  byte   0 to 232-1  1  N/A  9 - 10 
Router Memory Utilization    Integer*2  %  0 to 100  1  N/A  11 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  12 
CSU Loss of Signal  Number of times Loss of Signal event detected  Integer*4  N/A  0 to 232-1  1  N/A  13 - 14 
CSU Loss of Frames  Number of times Loss of Frames event detected  Integer*4  N/A  0 to 232-1  1  N/A  15 - 16 
CSU Yellow Alarms  Number of times Resource Availability Indication (RAI) (yellow) alarm received.  Integer*4  N/A  0 to 232-1  1  N/A  17 - 18 
CSU Blue Alarms  Number of times Alarm Indication Signal (AIS) (blue) alarm received.  Integer*4  N/A  0 to 232-1  1  N/A  19 - 20
CSU 24hr Errored Seconds  Number of errored seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  21 - 22 
CSU 24hr Severely Errored Seconds  Number of severely errored seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  23 - 24 
CSU 24hr Severely Errored Framing Seconds  Number of severely errored framing seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  25 - 26 
CSU 24hr Unavailable Seconds  Number of unavailable seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  27 - 28 
CSU 24hr Controlled Slip Seconds  Number of controlled slip seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  29 - 30 
CSU 24hr Path Coding Violations  Number of path coding violations in previous 24 hours.  Integer*4  N/A  0 to 232-1  1  N/A  31 - 32 
CSU 24hr Line Errored Seconds  Number of line errored seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  33 - 34 
CSU 24hr Bursty Errored Seconds  Number of bursty errored seconds in previous 24 hours.  Integer*4  s  0 to 232-1  1  N/A  35 - 36 
CSU 24hr Degraded Minutes  Number of degraded minutes in previous 24 hours.  Integer*4  min  0 to 232-1  1  N/A  37 - 38 
LAN Switch Memory Used  Bytes currently in use by applications on this device  Integer*4  byte  0 to 232-1  1  N/A  39 - 40
LAN Switch Memory Free  Bytes currently free on this device  Integer*4  byte  0 to 232-1  1  N/A  41 - 42 
LAN Switch Memory Utilization    Integer*2  %  0 to 100  1  N/A  43 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  44 
NTP Rejected Packets  Number of packets rejected by NTP application layer  Integer*4  N/A  0 to 232-1  1  N/A  45 - 46 
NTP Estimated Time Error  Current estimated time error of the time server  SInteger*4  usec  -(231) to +(231-1)  1  N/A  47 - 48 
GPS Satellites  Current number of GPS satellites used in position and time fix calculation  SInteger*4  N/A  -(231) to +(231-1)  1  N/A  49 - 50 
GPS Max Signal Strength  Strongest signal strength of all tracking satellites as seen by receiver  SInteger*4  dB  -(231) to +(231-1)  1  N/A  51 - 52 
IPC Status  Status of the communications between channels on a redundant system. N/A on a Single channel system.  Integer*2  N/A  0 to 2  1  0=OK, 1=Fail, 2=N/A  53 
Commanded Channel Control  Indicates which channel the RDA has commanded to be the controlling channel. This is not necessarily the channel which is in control.  Integer*2  N/A  0 to 2   1  0=N/A,  1=Channel 1,  2=Channel 2  54 
DAU Test 0  Tests the performance of the DAU A/D Mutiplexer with a known low voltage input.  Integer*2  N/A  0 to 255  1  10 = Normal, 7-11 = Acceptable, All other values= Fault  55 
DAU Test 1  Tests the performance of the DAU A/D Mutiplexer with a known medium voltage input.  Integer*2  N/A  0 to  255   1  127 = Normal, 118-136 = Acceptable, All other values=Fault  56 
DAU Test 2  Tests the performance of the DAU A/D Mutiplexer with a known high voltage input.  Integer*2  N/A  0 to  255   1  245 = Normal, 221-252 = Acceptable, All other values=Fault  57 
AME Polarization    Integer*2  N/A  0 to 2  1  0 = H Only 1 = H + V 2 = V Only  58 
AME Internal Temperature    Real*4  deg C  -40.0 to +125.0  0.1 N/A  59-60 
AME Receiver Module Temperature    Real*4  deg C  -40.0 to +125.0  0.1 N/A  61-62 
AME BITE/CAL Module Temperature    Real*4  deg C  -40.0 to +125.0  0.1 N/A  63-64 
AME Peltier Pulse Width Modulation    Integer*2  %  0 to 100  1  N/A  65 
AME Peltier Status    Integer*2  N/A  0 to 1  1  0 = OFF 1 = ON  66 
AME A/D Converter Status    Integer*2  N/A  0 to 1  1  0 = OK 1 = FAIL  67 
AME State    Integer*2  N/A  0 to 3  1  0 = START 1 = RUNNING 2 = FLASH 3 = ERROR  68 
AME +3.3V PS Voltage    Real*4  V  0.00 to 4.09  0.01 N/A  69-70
AME +5V PS Voltage    Real*4  V  0.00 to 6.10  0.01 N/A  71-72 
AME +6.5V PS Voltage    Real*4  V  0.00 to 7.50  0.01 N/A  73-74 
AME +15V PS Voltage    Real*4  V  0.00 to 19.00  0.01 N/A  75-76 
AME +48V PS Voltage    Real*4  V  0.00 to 60.00  0.01 N/A  77-78 
AME STALO Power    Real*4  V  0.00 to 4.09  0.01 N/A  79-80 
Peltier Current    Real*4  A  0.00 to 16.00  0.01 N/A  81-82 
ADC Calibration Reference Voltage    Real*4  V  0.000 to 2.048  0.001  N/A  83-84 
AME Mode    Integer*2  N/A  0 to 1  1  0 = READY 1 = MAINTENANCE  85 
AME Peltier Mode    Integer*2  N/A  0 to 1  1  0 = COOL 1 = HEAT  86 
AME Peltier Inside Fan Current    Real*4  A  0.00 to 4.00  0.01 N/A  87-88 
AME Peltier Outside Fan Current    Real*4  A  0.00 to 4.00  0.01 N/A  89-90 
Horizontal TR Limiter Voltage    Real*4  V  0.00 to 5.00  0.01 N/A  91-92 
Vertical TR Limiter Voltage    Real*4  V  0.00 to 5.00  0.01 N/A  93-94 
ADC Calibration Offset Voltage    Real*4  mV  -50.000 to +50.000  0.01 N/A  95-96 
ADC Calibration Gain Correction    Real*4  N/A  0.990 to 1.010  0.001  N/A  97-98 
Power UPS Battery Status    Integer*4  N/A  1 to 3  1  1=Unknown,  2=OK,  3=Low  99 - 100 
UPS Time on Battery    Integer*4  s  0 to 232-1  1  N/A  101 - 102
UPS Battery Temperature    Real*4  deg C  N/A  0.01  N/A  103 - 104 
UPS Output Voltage    Real*4  V  114.00 to 126.00  0.01  N/A  105 - 106 
UPS Output Frequency    Real*4  Hz  57.00 to 63.00  0.01  N/A  107 - 108 
UPS Output Current    Real*4  A  0.00 to 12.00  0.01  N/A  109 - 110 
Power Administrator Load    Real*4  A  0.00 to 12.00  0.01  N/A  111 - 112 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  113 - 136 
+5 VDC PS    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  137
+15 VDC PS    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  138 
+28 VDC PS    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  139
-15 VDC PS    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  140 
+45 VDC PS    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  141
Filament PS Voltage    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  142
Vacuum Pump PS Voltage    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  143 
Focus Coil PS Voltage    Integer*2  N/A  0 to 1  1  0=OK,1=Fail  144 
Filament PS    Integer*2  N/A  0 to 1  1  0=On, 1=Off  145
Klystron Warmup    Integer*2  N/A  0 to 1  1  0=Normal, 1=Preheat  146
Transmitter Available    Integer*2  N/A  0 to 1  1  0=Yes, 1=No  147
WG Switch Position    Integer*2  N/A  0 to 1  1  0=Antenna,  1=Dummy Load  148 
WG/PFN Transfer Interlock    Integer*2  N/A  0 to 1  1  0=OK, 1=Open  149 
Maintenance Mode    Integer*2  N/A  0 to 1  1  0= No, 1=Yes  150 
Maintenance Required    Integer*2  N/A  0 to 1  1  0=No,  1=Required  151 
PFN Switch Position    Integer*2  N/A  0 to 1  1   0=Short Pulse, 1=Long Pulse  152 
Modulator Overload    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  153 
Modulator Inv Current    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  154
Modulator Switch Fail    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  155 
Main Power Voltage    Integer*2  N/A  0 to 1  1  0=OK, 1=Over  156 
Charging System Fail    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  157 
Inverse Diode Current    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  158 
Trigger Amplifier    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  159 
Circulator Temperature    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  160 
Spectrum Filter Pressure    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  161 
WG ARC/VSWR    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  162 
Cabinet Interlock    Integer*2  N/A  0 to 1  1  0=OK, 1=Open  163 
Cabinet Air Temperature    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  164 
Cabinet Airflow    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  165 
Klystron Current    Integer*2  N/A   0 to 1   1  N/A  166 
Klystron Filament Current    Integer*2  N/A   0 to 1   1  0=OK, 1=Fail  167 
Klystron Vacion Current    Integer*2  N/A   0 to 1   1  0=OK, 1=Fail  168 
Klystron Air Temperature    Integer*2  N/A   0 to 1   1  0=OK, 1=Fail  169 
Klystron Airflow    Integer*2  N/A   0 to 1  1  0=OK, 1=Fail  170 
Modulator Switch Maintenance    Integer*2  N/A  0 to 1   1  0=OK,  1=Required  171 
Post Charge Regulator Maintenance    Integer*2  N/A   0 to 1   1  0=OK,  1=Maintenance  172 
WG Pressure/Humidity    Integer*2  N/A   0 to 1   1  0=OK, 1=Fail  173 
Transmitter Overvoltage    Integer*2  N/A  0 to 1  1  0=OK, 1=Over  174
Transmitter Overcurrent    Integer*2  N/A  0 to 1  1  0=OK, 1=Over  175 
Focus Coil Current    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  176 
Focus Coil Airflow    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  177 
Oil Temperature    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  178 
PRF Limit    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  179 
Transmitter Oil Level    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  180 
Transmitter Battery Charging    Integer*2  N/A  0 to 1  1  0=Yes, 1=No  181 
High Voltage (HV) Status    Integer*2  N/A    0 to 1   1  0=On, 1=Off  182 
Transmitter Recycling Summary    Integer*2  N/A   0 to 1  1  0=Normal, 1=Recycling  183 
Transmitter Inoperable    Integer*2  N/A  0 to 1  1  0=OK, 1=INOP  184 
Transmitter Air Filter    Integer*2  N/A   0 to 1  1  0=Dirty, 1=OK  185 
Zero Test Bit 0    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  186 
Zero Test Bit 1    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  187 
Zero Test Bit 2    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  188 
Zero Test Bit 3    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  189 
Zero Test Bit 4    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  190 
Zero Test Bit 5    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  191 
Zero Test Bit 6    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  192 
Zero Test Bit 7    Integer*2  N/A  0 to 1   1  0=OK, 1=Fail  193 
One Test Bit 0    Integer*2  N/A   0 to 1   1  0=Fail, 1=OK  194 
One Test Bit 1    Integer*2  N/A   0 to 1   1   0=Fail, 1=OK  195 
One Test Bit 2    Integer*2  N/A   0 to 1   1  0=Fail, 1=OK  196 
One Test Bit 3    Integer*2  N/A   0 to 1   1  0=Fail, 1=OK  197 
One Test Bit 4    Integer*2  N/A  0 to 1   1  0=Fail, 1=OK  198 
One Test Bit 5    Integer*2  N/A  0 to 1   1  0=Fail, 1=OK  199 
One Test Bit 6    Integer*2  N/A  0 to 1   1  0=Fail, 1=OK  200 
One Test Bit 7    Integer*2  N/A    0 to 1   1   0=Fail, 1=OK  201 
XMTR/DAU Interface    Integer*2  N/A  0 to 1  1  0=Fail, 1=OK  202 
Transmitter Summary Status    Integer*2  N/A  0 to 4  1  0=Ready, 1=Alarm, 2=Maintenance, 3=Recycle, 4=Preheat  203 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  204 
Transmitter RF Power (Sensor)    Real*4  mW  0.0000 to 10.0000  .0001  N/A   205 - 206 
Horizontal XMTR Peak Power    Real*4  kW  0 to 999.9  0.1  N/A  207 - 208 
XMTR Peak Power    Real*4  kW  0 to 999.9  0.1  N/A  209 - 210 
Vertical XMTR Peak Power    Real*4  kW  0 to 999.9  0.1  N/A  211 - 212 
XMTR RF Avg Power    Real*4  W  0 to 9999.9  0.1  N/A  213 - 214 
XMTR Power Meter Zero    Integer*2  N/A  0 to 255  1  N/A  215 
Spare    N/A  N/A  N/A  N/A  See Note (3)  216 
XMTR Recycle Count    Integer*4  N/A  0 to  999,999  1  N/A  217 - 218 
Receiver Bias    Real*4  dB  0 to  999.9999  0.0001  N/A  219 - 220
Transmit Imbalance    Real*4  dB  0 to 999.99  0.01  N/A  221 - 222
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  223 - 228
AC Unit #1 Compressor Shut off    Integer*2  N/A  0 to 1  1  0=OK, 1=Shutoff  229 
AC Unit #2 Compressor Shut off    Integer*2  N/A  0 to 1  1  0=OK, 1=Shutoff  230 
Generator Maintenance Required    Integer*2  N/A  0 to 1  1  0=Yes, 1=No  231 
Generator Battery Voltage    Integer*2  N/A  0 to 1  1  0=Low, 1= OK  232 
Generator Engine    Integer*2  N/A  0 to 1  1  0=Fail, 1=OK  233 
Generator Volt/Frequency    Integer*2  N/A  0 to 1  1  0=Not available, 1=Available  234 
Power Source    Integer*2  N/A  0 to 1  1  0=Utility Power, 1=Generator Power  235 
Transitional Power Source (TPS)    Integer*2  N/A  0 to 1  1  0=OK, 1=Off  236 
Generator Auto/Run/Off Switch    Integer*2  N/A  0 to 1  1  0=Manual, 1=Auto  237 
Aircraft Hazard Lighting    Integer*2  N/A  0 to 1  1  0=Fail, 1=OK  238 
DAU UART    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  239 
Spare  N/A  N/A  N/A  N/A  1  See Note (3)  240 - 249 
Equipment Shelter Fire Detection System    Integer*2  N/A  0 to 1  1  0 = OK, 1 = Fail  250 
Equipment Shelter Fire/Smoke    Integer*2  N/A  0 to 1  1  0=OK, 1=Fire  251 
Generator Shelter Fire/Smoke    Integer*2  N/A  0 to 1  1  0=Fire, 1=OK  252 
Utility Voltage/Frequency    Integer*2  N/A  0 to 1  1  0=Not available, 1=Available  253 
Site Security Alarm    Integer*2  N/A  0 to 1  1  0=Alarm, 1=OK  254 
Security Equipment    Integer*2  N/A  0 to 1  1  0=Fail, 1=OK  255 
Security System    Integer*2  N/A  0 to 1  1  0=Disabled,  1=OK  256 
Receiver Connected to Antenna    Integer*2  N/A  0 to 2  1  N/A on a single channel system. 0=Connected, 1=Not Connected, 2=N/A  257 
Radome Hatch    Integer*2  N/A  0 to 1  1  0=Open,  1=Closed  258 
AC Unit #1 Filter Dirty    Integer*2  N/A  0 to 1  1  0=Dirty, 1=OK  259 
AC Unit #2 Filter Dirty    Integer*2  N/A  0 to 1   N/A  0=Dirty, 1=OK  260 
Equipment Shelter Temperature    Real*4  deg C  0.00 to +50.00  0.01  N/A  261 - 262 
Outside Ambient Temperature    Real*4  deg C  -50.00 to +50.00  0.01  N/A  263 - 264 
Transmitter Leaving Air Temp    Real*4  deg C  -10.00 to +60.00  0.01  N/A  265 - 266 
AC Unit #1 Discharge Air Temp    Real*4  deg C  0.00 to +50.00  0.01  N/A  267 - 268 
Generator Shelter Temperature    Real*4  deg C  0.00 to +50.00  0.01  N/A  269 - 270 
Radome Air Temperature    Real*4  deg C  -50.00 to +50.00  0.01  N/A  271 - 272 
AC Unit #2 Discharge Air Temp    Real*4  deg C  0.00 to +50.00  0.01  N/A  273 - 274 
DAU +15v PS    Real*4  V  0.00 to 20.00   0.01  N/A  275 - 276 
DAU -15v PS    Real*4  V  -20.00 to 0.00  0.01  N/A  277 - 278 
DAU +28v PS    Real*4  V   0.00 to 37.40   0.01  N/A  279 - 280 
DAU +5v PS    Real*4  V   0.00 to 6.64   0.01  N/A  281 - 282 
Converted Generator Fuel Level    Integer*2  %  0 to 100  1  N/A  283 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  284 - 290 
+28v PS    Real*4  V  0.00  to  40.80  0.01  N/A  291 - 292
Pedestal +15v PS    Real*4  V  0.00  to  20.00  0.01  N/A  293 - 294 
Encoder +5v PS    Real*4  V  0.00  to 18.36  0.01  N/A  295 - 296 
Pedestal +5v PS    Real*4  V  0.00  to 6.64   0.01  N/A  297 - 298 
Pedestal -15v PS    Real*4  V  -20.00 to 0.00  0.01  N/A  299 - 300 
+150V Overvoltage    Integer*2  N/A  0 to 1   1  0=OK,  1=Overvoltage  301 
+150V Undervoltage    Integer*2  N/A  0 to 1  1  0=OK,  1=Overvoltage  302 
Elevation Servo Amp Inhibit    Integer*2  N/A  0 to 1  1  0=Normal, 1=Inhibit  303 
Elevation Servo Amp Short Circuit    Integer*2  N/A  0 to 1  1  0=Normal, 1=Short Circuit  304 
Elevation Servo Amp Overtemp    Integer*2  N/A  0 to 1  1  0=Normal,  1=Overtemp  305 
Elevation Motor Overtemp    Integer*2  N/A  0 to 1  1  0=OK,  1=Overtemp  306 
Elevation Stow Pin    Integer*2  N/A  0 to 1  1  0=Operational, 1=Engaged  307 
Elevation PCU Parity    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  308 
Elevation Dead Limit    Integer*2  N/A  0 to 1  1  0=OK,  1=In Limit  309 
Elevation +Normal Limit    Integer*2  N/A  0 to 1  1  0=OK,  1=In Limit  310 
Elevation -Normal Limit    Integer*2  N/A  0 to 1  1  0=OK,  1=In Limit  311 
Elevation Encoder Light    Integer*2  N/A  0 to 1  1  1=Fail, 0=OK  312 
Elevation Gearbox Oil    Integer*2  N/A  0 to 1  1  0=OK, 1=Oil Level Low  313
Elevation Handwheel    Integer*2  N/A  0 to 1  1  0=Operational, 1=Engaged  314 
Elevation Amp PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  315 
Azimuth Servo Amp Inhibit    Integer*2  N/A  0 to 1  1  0=OK,  1=Inhibit  316 
Azimuth Servo Amp Short Circuit    Integer*2  N/A  0 to 1  1  1=Short Circuit, 0=OK  317 
Azimuth Servo Amp Overtemp    Integer*2  N/A  0 to 1  1  0=OK, 1=Overtemp  318 
Azimuth Motor Overtemp    Integer*2  N/A  0 to 1  1  0=OK, 1=Overtemp  319 
Azimuth Stow Pin    Integer*2  N/A  0 to 1  1  0=Operational, 1=Engaged  320 
Azimuth PCU Parity    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  321 
Azimuth Encoder Light    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  322 
Azimuth Gearbox Oil    Integer*2  N/A  0 to 1  1  0=OK, 1=Oil Level Low  323 
Azimuth Bull Gear Oil    Integer*2  N/A  0 to 1  1  0=OK, 1=Oil Level Low  324 
Azimuth Handwheel    Integer*2  N/A  0 to 1  1  0=Operational, 1=Engaged  325 
Azimuth Servo Amp PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  326 
Servo    Integer*2  N/A  0 to 1  1  0=On, 1=Off  327 
Pedestal Interlock Switch    Integer*2  N/A  0 to 1  1  0=Operational, 1=Safe  328 
Spare  N/A  N/A  N/A  N/A  N/A   See Note (3).  329 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3).  330 
Self Test 1 Status    Integer*2  N/A  1 to 3  1  1=No, 2=OK, 3=Fail See Note(1)  331
Self Test 2 Status    Integer*2  N/A  1 to 3  1  1=No, 2=OK, 3=Fail See Note(1)  332 
Self Test 2 Data    Integer*2  N/A  N/A  1  Hex See Note (2)  333 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  334 - 340 
COHO/Clock    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  341 
Rf Generator Frequency Select Oscillator    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  342 
Rf Generator RF/STALO    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  343 
Rf Generator Phase Shifted COHO    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  344 
+9v Receiver PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  345 
+5v Receiver PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  346 
O/18v Receiver PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  347 
-9v Receiver PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  348 
+5v Single Channel RDAIU PS    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  349 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  350 
Horizontal Short Pulse Noise    Real*4  dBm  -100.00 to -50.00  0.01  N/A  351 - 352 
Horizontal Long Pulse Noise    Real*4  dBm  -100.00 to -50.00  0.01  N/A  353 - 354 
Horizontal Noise Temperature    Real*4  K  0 to 9999.99   0.01  N/A  355 - 356 
Vertical Noise Short Pulse   N/A  N/A  N/A  N/A  N/A  See Note (3)  357 - 358 
Vertical Noise Long Pulse    Real*4  dBm  -100.00 to -50.00   0.01 N/A  359-360 
Vertical Noise Temperature    Real*4  K  0 to 9999.99  0.01 N/A  361-362
Horizontal Linearity    Real*4  N/A  0.5000 to 1.5000  0.0001  N/A  363 - 364 
Horizontal Dynamic Range    Real*4  dB  0.000 to 120.000  0.001  N/A  365 - 366 
Horizontal Delta dBZ0    Real*4  dB  -198.00 to +198.00  0.01  N/A  367 - 368 
Vertical Delta dBZ0    Real*4  dB  -198.00 to +198.00   0.01  N/A  369 - 370 
KD Peak Measured    Real*4  dBm  -99.90 to +99.90  0.01  N/A  371 - 372 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  373 - 374 
Short Pulse, Horizontal dBZ0    Real*4  dBZ  -99.900 to +99.900   0.0001  N/A  375 - 376 
Long Pulse, Horizontal dBZ0    Real*4  dBZ  -99.9000 to +99.9000  0.0001  N/A  377 - 378 
Velocity (Processed)    Integer*2  N/A  0 to 1  1  0=Good, 1=Fail  379 
Width (Processed)    Integer*2  N/A  0 to 1  1  0=Good, 1=Fail  380 
Velocity (RF Gen)    Integer*2  N/A  0 to 1  1  0=Good, 1=Fail  381 
Width (RF Gen)    Integer*2  N/A  0 to 1  1  0=Good, 1=Fail  382 
Horizontal I0    Real*4  dBm  -999.9000 to +999.9000  0.0001  N/A  383 - 384 
Vertical I0    Real*4  dBm  -999.9000 to +999.9000  0.0001  N/A  385 - 386 
Vertical Dynamic Range    Real*4  dB  0.000 to 120.000  0.001  N/A  387 - 388 
Short Pulse, Vertical dBZ0    Real*4  dBZ  -99.9000 to +99.9000  0.0001  N/A  389 - 390 
Long Pulse, Vertical dBZ0    Real*4  dBZ  -99.9000 to +99.9000  0.0001  N/A  391 - 392 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  393 - 394 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  395 - 396 
Horizontal Power Sense    Real*4  dBm  -999.9000 to +999.9000  0.0001  N/A  397 - 398
Vertical Power Sense    Real*4  dBm  -999.9000 to +999.9000  0.0001  N/A  399 - 400 
ZDR Bias    Real*4  dB  -999.9000 to +999.9000  0.0001  N/A  401 - 402 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  403 - 408 
Clutter Suppression Delta    Real*4  dB  -99.90 to +99.90  0.01 N/A  409-410 
Clutter Suppression Unfiltered Power    Real*4  dBZ  -99.90  to  +99.90  0.01  N/A  411 - 412 
Clutter Suppression Filtered Power    Real*4  dBZ  -99.90  to  +99.90   0.01  N/A  413 - 414 
Transmit Burst Power    Real*4  dBm  -99.90 to +99.90  0.01  N/A  415 - 416 
Transmit Burst Phase    Real*4  deg  -99.00 to +99.90  0.01  N/A  417 - 418 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  419 - 422 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  423 - 424 
Vertical Linearity    Real*4  N/A  0.5000 to 1.5000  0.0001  N/A  425 - 426 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  427 - 430 
State File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  431 
State File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  432 
Bypass Map File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  433 
Bypass Map File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  434 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  435 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  436 
Current Adaptation File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  437 
Current Adaptation File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  438 
Censor Zone File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  439 
Censor Zone File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  440 
Remote VCP File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  441 
Remote VCP File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  442 
Baseline Adaptation File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  443 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  444 
Clutter Filter Map File Read Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  445 
Clutter Filter Map File Write Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  446 
General Disk I/O Error    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  447 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  448 - 460 
DAU Comm Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  461 
HCI Comm Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  462 
Pedestal Comm Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  463 
Signal Processor Comm Status    Integer*2  N/A  0 to 1  1  0=OK, 1=Fail  464 
AME Communication Status    Integer*2  N/A  0 to 1  1  0 = OK 1 = FAIL  465 
RMS Link Status    Integer*2  N/A  0 to 1  1  0 = Connected, 1 = Not Connected  466 
RPG Link Status    Integer*2  N/A  0 to 1  1  0 = Connected, 1 = Not Connected  467 
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  468
Performance Check Time    Integer*4  N/A  N/A  1    469 - 470
Spare  N/A  N/A  N/A  N/A  N/A  See Note (3)  471 - 480
