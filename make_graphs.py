import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from bs4 import BeautifulSoup
import cPickle as pickle
from bokeh.charts import Bar, output_file, show
from unidecode import unidecode

from sklearn.utils import resample

bad_payee_name = set(['',
'48BiltmoreAvenue,LLCdbaChestnut',
'AIGAMetro-North',
'AlexandriaMarshall',
'AnnFranzen',
'AnnieZurth',
'BarierSpiziauLana',
'C.Cornish',
'ChavezFamilyVision',
'ColumbiaBusinessClubofWashingtonc/oKowitt',
'DavidAkasa',
'DavidBenjamin',
'EileenaKim',
'FrancisHernandez',
'FriendsofTheAllenSchool',
'GaryL.Johnson',
'GlassCactusEntertainment',
'GlennAlan',
'GlobalGasCard',
'GuadalupeCulturalArtsCenter',
'IsabelGayKelly',
'JasonPhillips',
'Jeremywilliams',
'JeronKnox',
'JohnMarcusCabrera',
'KyleWoods',
'LiveOakMBC',
'MariaHoatson',
'NA',
'NicoleMeans',
'OrlandoHernandez',
'PARTYSTARZENTINC',
'RoyalPrincessesandKnights',
'SarahSobieski',
'Seamar',
'StevenKadlec',
'TaquanHenderson',
'TimGlass',
'robertcarter',
'x'])

bad_domain = set(['126.com',
'19sieunhan.com',
'31and7.com',
'4asdkids.com',
'4u2nv-ent.com',
'9and1.biz',
'DionJordan.com',
'GMAIL.COM',
'NA',
'REDCLAYSOULNOLA.COM',
'Safe-mail.net',
'The2Half.com',
'accountant.com',
'actingscore.com',
'alhambrapalacerestaurant.com',
'anasconcept.com',
'angelwish.org',
'aol.co.uk',
'aol.com',
'ashfordradtech.org',
'att.net',
'betaltd.org',
'blader.com',
'bravuraconsulting.com',
'brew-master.com',
'brew-meister.com',
'bww.com',
'cannapro.com',
'catchatt.com',
'cbsdcalumclub.com',
'cdrenterprise.net',
'certifiedforensicloanauditors.com',
'chavezfv.org',
'checker.vn',
'chef.net',
'clerk.com',
'clothmode.com',
'comcast.net',
'consultant.com',
'contractor.net',
'cox.net',
'cs.com',
'ctq.net.au',
'cyberservices.com',
'datachieve.com',
'dcbtf.org',
'discofan.com',
'diversity-church.com',
'dr.com',
'emgay.com',
'eng.uk.com',
'europe.com',
'execs.com',
'freya.pw',
'fridayzonmarz.co.uk',
'frontier.com',
'gawab.com',
'gcase.org',
'ger-nis.com',
'girl-geeks.co.uk',
'gmail.com',
'gmx.com',
'googlemail.com',
'gosimplysocial.com',
'greenrcs.com',
'guadalupeculturalarts.org',
'hamptonmedi.com',
'heresyemail.com',
'hmshost.com',
'hotelvenizbaguio.com',
'hotmail.co.uk',
'hotmail.com',
'hotmail.de',
'hotmail.fr',
'hushmail.com',
'idealpublicity.com',
'in.com',
'inbox.com',
'indglobal-consulting.com',
'indiabestplace.com',
'inkfestlive.com',
'innovateyours.com',
'inorbit.com',
'instructor.net',
'insuranceadjustersinc.com',
'investocorp.com',
'ioccupied.net',
'izzane.net',
'jcclain.com',
'jobsfc.com',
'kbzaverigroup.com',
'keromail.com',
'ladiesat11.com',
'launchpadinw.com',
'leisurelodgebaguio.com',
'levyresourcesinc.com',
'lidf.co.uk',
'live.FR',
'live.ca',
'live.co.uk',
'live.com',
'live.de',
'live.fr',
'lmtexformula.com',
'london.com',
'lovetheneighbor.info',
'lushsaturdays.co.uk',
'mail.com',
'maroclancers.com',
'me.com',
'medicalrepinsight.com',
'mohmal.com',
'monkeyadvert.com',
'msn.com',
'myself.com',
'myway.com',
'nantucketfilmfestival.org',
'naworld-x.com',
'nbuux.com',
'noiphuongxa.com',
'outlook.com',
'outlook.de',
'outlook.fr',
'ovidcapita.com',
'petlover.com',
'photographer.net',
'post.com',
'premier3.com',
'presinnercircle.com',
'pridetoronto.com',
'primehire.co.uk',
'qip.ru',
'qualityservice.com',
'quaychicago.com',
'ravemail.com',
'republicanflorida.com',
'rock.com',
'rocketmail.com',
'safe-mail.net',
'smokinbettys.com',
'socialsolutionsacademy.co.uk',
'socialworker.net',
'startupmaroc.com',
'student.framingham.edu',
'techie.com',
'themoonbridge.com',
'theparadigmcollective.com',
'thickdame.com',
'thinktankconsultancy.com',
'toke.com',
'twcny.rr.com',
'ultimatewine.co.uk',
'usa.com',
'visichathosting.net',
'vncall.net',
'yahoo.ca',
'yahoo.co.id',
'yahoo.co.uk',
'yahoo.com',
'yahoo.com.ar',
'yahoo.com.vn',
'yahoo.de',
'yahoo.fr',
'yahoo.it',
'yellamo.com',
'ymail.com',
'yopmail.com',
'zumba-perth.com'])

bad_venue_state = set(['',
'AK',
'AL',
'AR',
'AZ',
'AbuDhabi',
'Algiers',
'BA',
'Bali',
'Bedfordshire',
'Belfast',
'Berlin',
'Birmingham',
'Blackpool',
'Bristol,CityOf',
'Burgandy',
'CA',
'CAR',
'CO',
'CT',
'Cambs',
'CapitalRegionofDenmark',
'Cardiff',
'CentralJava',
'Centre',
'Ceredigion',
'CityofEdinburgh',
'CityofZagreb',
'CordilleraAdministrativeRegion',
'Coventry',
'Cumbria',
'Cundinamarca',
'DC',
'DE',
'Derry',
'Doukkala-Abda',
'Dubai',
'England',
'Erongo',
'Essex',
'FL',
'Famagusta',
'Faro',
'Florida',
'GA',
'GP',
'GlasgowCity',
'GrandCasablanca',
'GreaterLondon',
'GtLon',
'Guanajuato',
'HE',
'HI',
'Hampshire',
'Hanoi',
'Hertfordshire',
'HoChiMinhCity',
'IDF',
'IL',
'IN',
'IdF',
'IleDeFrance',
'Islington',
'Istanbul',
'Jakarta',
'KY',
'KarachiDistrict',
'KhanhHoaprovince',
'Khomas',
'KingstonuponHull',
'LA',
'Lagos',
'Lancashire',
'Languedoc-Roussillon',
'Lazio',
'Leicester',
'London',
'London,CityOf',
'LondonBoroughofEnfield',
'Luton',
'MA',
'MD',
'MI',
'MN',
'MO',
'MT',
'Manchester',
'Marrakesh-Tensift-AlHaouz',
'Middlesex',
'MiltonKeynes',
'Mt',
'Muscat',
'NA',
'NC',
'NCR',
'NDS',
'NE',
'NEWSOUTHWALES',
'NH',
'NJ',
'NM',
'NRW',
'NSW',
'NV',
'NY',
'Nairobi',
'Nakuru',
'NewcastleUponTyne',
'Northamptonshire',
'OH',
'OK',
'ON',
'OR',
'Ontario',
'PA',
'PCh',
'PaysDeLaLoire',
'PerthAndKinross',
'PhnomPenh',
'Punjab',
'QC',
'QLD',
'QuangBinhprovince',
'RI',
'Ranong',
'Reading',
'RoyalBoroughofKensingtonandChelsea',
'SA',
'SC',
'Sindh',
'Solihull',
'Souss-Massa-Draa',
'SouthAyrshire',
'SouthYork',
'StHelier',
'StThomas',
'Stockholmslan',
'Surrey',
'Swansea',
'TAS',
'TN',
'TX',
'Tangier-Tetouan',
'TienGiang',
'UKN',
'UT',
'VA',
'VIC',
'Victoria',
'Voronezhskayaoblast',
'WA',
'WI',
'Warks',
'Warwickshire',
'WestJava',
'WestMids',
'Wokingham',
'Wolverhampton'])

bad_venue_country=set(['',
'AE',
'AR',
'AU',
'CA',
'CM',
'CO',
'CY',
'DE',
'DK',
'DZ',
'FR',
'GB',
'HR',
'ID',
'IT',
'JE',
'KE',
'KH',
'MA',
'MX',
'NA',
'NG',
'NL',
'None',
'OM',
'PH',
'PK',
'PT',
'RU',
'SE',
'TH',
'TR',
'US',
'VI',
'VN',
'ZA'])

bad_venue_address = set(['',
'.1509&1517HAve',
'1-7ChapelStreetsM37NJ',
'10',
'1000ChampionsDrive',
'10010thAvenue',
'1001WestColumbusAve',
'1002SouthBarnettSt',
'1008VermontAvenueNorthwest',
'100E.PennSquare',
'100WarburgerStrasse',
'10110GlenRoseCourt',
'10130PerimeterParkway',
'101GlenhuntlyRoad',
'101TempleWay',
'10236CharingCrossRoad',
'102PeterSt',
'10307N.MallDr.',
'1039WashingtonStreet',
'104,AlgernonRoad',
'10401WytonDrive',
'10440InternationalBlvd',
'10455thstreet',
'1050CenturyDr',
'10555thavenue',
'105WinterberryRidge',
'1065PeachtreeStreetNE',
'106ChickadeeLane',
'106PeterSt',
'10715thAve',
'1075BiscayneBlvd',
'10768SeacliffCircle',
'10909Mst#170',
'10911WestheimerRd',
'10989FrankstownAve',
'10Aldermanbury',
'10ColumbusCircle',
'10DrewAvenue',
'10MargaretStree',
'10RooneyAve.',
'10SouthWaterStreet',
'1101GreentreeCt',
'1101WilsonBlvd',
'110UniversityPlace,NewYorkNY10003',
'111114thStreet',
'111E.KIRBYST.',
'111E48thSt,NewYork,NY10017',
'111MinnaStreet',
'1129ValleyRd',
'112N5thSt',
'112TravisStreet',
'1150CampHillbypass',
'11610thAve',
'117SouthMatthisenAvenue',
'1185AvenueoftheAmericas,NewYorkNY',
'119EdwardStreet',
'119LondonRd',
'11BeauchampPl',
'11avenuefoch',
'12-15OsbornStreet',
'1200EpcotResortsBlvd',
'1200SFrenchAve',
'120N.Stevens',
'1214QueenStW',
'122StMarksPl#126',
'12330NGessnerDr',
'12389PEMBROKEROAD',
'1239EASTLASOLASBLVD',
'1240WRandolphSt',
'124W42ndSt',
'125E.11thStreet',
'125East11thstreet3rdand4thave',
'125W.45thSt(btwn6thand7thAve)',
'127IdahoDr',
'128-132BoroughHighStreet',
'1280PeachtreeStNE',
'12873StageCoachDrive',
'1289S.DixieHwy',
'128NguyenKhanhToan',
'12955OldMeridianStCarmel,IN46032',
'129E15thSt,NewYorkNY',
'12ApBac',
'12EldridgeSt',
'12LeLoi',
"12Platt'sLn",
'130SCameronStreet',
'131West128thStreet',
'13331KuykendahlRd',
'1337WConnecticutSt',
'135FinchleyRoad',
'137CityviewDr',
'137WarrenAvenue',
'13SpraySt',
'1400SOUTHLINCOLNPARKWAY',
'14106NIH35',
'14108StPaulRoad',
'14119PearTreeLane,#34',
'1433CaminoDelRioSouth',
'144holloway,CityOfLondon',
'145-157',
'145East50thStreet',
'145SouthSaintAndrewsStreet',
'1461BaltimoreAnnapolisBlvd',
'146PRAEDSTREET',
'146PraedStreet',
'14800SeventhStreet',
'148HollowayRoad',
'1501CanyonDelRey',
'150BrickLane',
'150BrittainLn',
'1516PeachtreeStNE',
'151ClevelandStreet',
'151MurdockRd',
'151OldStreet',
'15203KnollTrailDr.',
'152WTamarackCir',
'1550LansdowneStW',
'15575JimmyDuranteBlvd',
'155SansomeStreet',
'15909PrestonRd',
'15WattsStreet,ManhattanNY',
'15WestEagerStreet',
'1601ClovisAve',
'1601CollinsAve',
'1610LakeviewDrive',
'161stAve',
'1630GreenwoodAvenue',
'163UnionSt',
'169plashetrd',
'16FirstAvenue',
'16W29thSt',
'16thStreetMall-DowntownDenver',
'1700ArmyNavyDrive',
'1700LakeKingwoodTrail',
'17011NE19THAVE',
'1701QueenStE',
'170West233rdStreet',
'1712WestPrattStreet',
'1716BushBlvdW',
'1716LakeShoreBlvdE',
'172WandsworthRoad',
'1753PantherValleyRoad',
'176N.BEACHST.',
'17723',
'177WATSONSMILLRD',
'17800NE5TH',
'17800NE5THAVE',
'17Crosswall',
'17RoxwellRd',
'17VinylCourt',
'18-22FinchleyRoad',
'1800MarketStreet',
'1814harrisonSt',
'18181stAve,Dallas,TX75210',
'1819uticaace',
'184StokeNewingtonHighStreet',
'1880S.DairyAshford',
'18KendallCourt',
'18LittleWest12thSt',
'18YorkvilleAvenue',
'19-21GreatMarlboroughStreet',
'1900KStreetNW',
'1901MississippiAvenue,SE',
'1917thAve.NewYork,NY10011',
'19239U.S.27',
'192PembrokeStreet',
'196ChasepointeDrive',
'1979Hwy35',
'1ACamdenHighStreet',
'1AllSaintsPassage',
'1CountryClubLn',
'1E161stSt',
'1HamiltonPl',
'1KensingtonHighSt',
'1LovatLn',
'1Mohrenstrasse',
'1PolitoAve',
'1ThompsonPark',
'1UniversityPl',
'1W53rdSt',
'1WeirRoad',
'1clareroad',
'1plattslane',
'200AdvanceBlvd',
'200ConventionSt.',
'200DohertyDr',
'200SPineAve',
'200SeaportBoulevard',
'201ParkAvenueSouth',
'2020SOrangeBlossomTrail',
'2020WPensacolaSt',
'202HagleyRd',
'203HollowayRoad',
'203NGeneseeSt',
'2040StCharlesAve',
'2052ENorthernLightsBlvd',
'205W46thSt',
'2079E.15THST.',
'209W5thStreet',
'20ChurchStreet',
'20West39thStreet,NewYorkNY',
'2100SHiawasseeRd',
'2109S.WABASH',
'216wplumstreet',
'21AlgoresWay',
'2201SouthOceanBlvd',
'220WChicagoAve',
'2225NorthLoisAve',
'2245SanoraDrive',
'224PiccadillyCircus',
'2255GladesRoad',
'225EdgwareRoad',
'225MarshWall',
'2278FirstSt',
'2289CedarStreet',
'22S23rdSt',
'2308AdamClaytonPowellJuniorBlvd.',
'2337ConeyIslandAve',
'238GainesAve',
'2400OldLincolnHwy',
'240BeachDrNE',
'240West52ndStreet',
'2430FDRDr',
'24408EmilyDrive',
'247West30thstreetbtwn7&8ave',
'24OldGloucesterStreet',
'2500ESecondStreet',
'250DuttonAve',
'2519HighPointRd',
'251HighStreet',
'251MontgomeryStreet',
'251buteterrace',
'2520MiamiRoad',
'2525BruenLane',
'255BiscayneBoulevardWay',
'25LittleWest12thStreetNewYork',
'25SouthQueenStreet',
'25West52ndStreet',
'2606HodgenvilleRoad,Elizabethtown,KY42701',
'260ChapmanRd.',
'260So.CaliforniaAve.',
'262OldStreet',
'262PharrRoadNortheast',
'2667PITKINAVENUE,SUITEB',
'268BMammothRoad',
'26SmithfieldStreet',
'2700NW167ST',
'275BakerSt.',
'27LocustSt',
'27RueduCheminRouge,',
'27sbanksst.',
'2801W.WestWacoDrive',
'28220JeffersonAvenue',
'2831Boardwalk',
'28NelsonStreet',
'291SunshineRd,Tottenham,VIC3012',
'292-294KenningtonRoad',
'29AtlanticAvenue',
'29EldonRoad',
'29MIDLANDROAD',
'29WoodbineDownsBlvd',
'2TownleyRd',
'2WillowRoad',
'3&4DeansgateLocks',
'3-4CoventryStreet',
'300EastWigwamBlvd.',
'300HarborviewPlaza',
'300Southwest1stAvenue',
'300cooperst',
'301Pfaffenwiese',
'30205Southwest217thAvenue',
'3025WalnutStreet',
'3035SEMaricampRd',
'3049ScottFutrellDr',
'3051RevereAve',
'308W.46thSt.',
'30Colonnade',
'30NewJersey156',
'3122N.LovingtonHwy.',
'312W.34thStreet,NewYorkNY10001',
'313SouthPineAvenue',
'3150WLincolnAve',
'3186SoundAvenue',
'31OldGloucesterStreet',
'32-1037thAve',
'3220ButnerRd',
'3231FillmoreSt',
'3251NFederalHighway',
'3280PeachtreeRdNE',
'32SmithSquare',
'3301AnnapolisRoad',
'3309BunkerHillRoad',
'333PoydrasStreet',
'333W4thAve',
'3365CheesequakeRoad',
'3368PeachtreeRdNE',
'337NE170ST',
'33RolandGardens',
'3415ZILLIAHST',
'3421East96Th.Street',
'342East5thAvenue',
'3433NorthLumpkinRoad',
'348thAve',
'34LombardRoad',
'34TaksimYaghanesiSokak',
'3501NorthBroadway',
'350S.DuvalStreet',
'351RiverdaleDr',
'352SNovaRd',
'3536CountryClubRd',
'3550CellarDoorWay',
'3570LasVegasBlvdSouth',
'35South2ndStreet',
'360bayresroad',
'361MetropolitanAve',
'3667SouthLasVegasBoulevard',
'368AtlanticAvenue',
'3700WFLAMINGOAVE',
'370HughesCenterDr.',
'372ValleyParkDr',
'3770LasVegasBlvd.',
'3775HedgeLane',
'37PockettsWharf',
'37PockettsWharf,',
'37west17thstreet',
'38500Hwy12',
'38614thAvenue#74',
'389BroomeStreet',
'3900WestGirardAvenue',
'390SOrangeAve',
'3987SocoRd',
'3EaglePointRoad',
'3TalismanRise',
'3rdFloor',
'4001WChurchSt',
'4002WChurchSt',
'400BinksForestDrive',
'400FifthAvenue',
'400W.CHURCHST.',
'400WChurchSt',
'401NHowardSt',
'402OrchardHillDr',
'402WChurchSt',
'40360Finley',
'4039NW16thBlvd',
'403E.FrontSt.',
'404euclidave',
'4085HIGHBRIDGERD.',
'409East59thStreet,NewYorkNY',
'409W13thStreet,NewYorkNY',
'409W14thStreet,NewYorkNY',
'409West14thStreet,NewYorkNY',
'40BayStreet',
'4100BlueMoundRd',
'4100WekivaClubCourt',
'410CAMPUSCENTERDRIVE',
'411NorthNewRiverDriveEast',
'41PleasantSt',
'421NorthAve19',
'4231AvenueoftheRepublic',
'4242CampusDrive',
'424NBeverlyDr',
'42ndStreet,NewYorkNY',
'4321WESTFLAMINGOROAD',
'433SouthPalmettoAve.',
'434LafayetteSt',
'43GordonSquare',
'441GraceAve',
'441WestBeachDrive',
'44East32ndStreet',
'450NorthCityfrontPlazaDrive',
'4525CollinsAvenue',
'454West128thSt',
'4551OldAirportRoad',
'455BercutDr',
'4567NorthLincolnAvenue',
'45DeanStreet',
'45MartinTerrace',
'46-50OLDHAM',
'463COURTSTREET',
'465EIllinoisSt',
'46900MissionBoulevard',
'4746NRacineAve',
'4755CastletonWay',
'475EMorelandAve',
'475MorelandAveSE',
'475WMorelandAve',
'47LillieRoad',
'4801EFowlerAve',
'480RueLessard',
'4815Hwy6',
'4829W77thSt',
'4848ConstitutionAvenue',
'4849WestIllinoisAve',
'48BiltmoreAve',
'48East23rdStreet,NewYorkNY10010',
'48WallStreet',
'4980DixonStreet',
'49SherwoodTerrace',
'4Bachstrasse',
'4PlaceVilleMarie',
'4RichmondMews',
'4THANDISLAND',
'500E30thSt,NewYork,NY10016',
'5010OldNationalHighway',
'501BroadwaySt.',
'50BernersStreet',
'50potteryroad',
'510Fairgroundsplace',
'515SouthRailroadBlvd',
'515W97thSt',
'526EHospitalSt',
'540PresidentStreetSuite2E',
'5441almedaave',
'54West21stStreet,NewYorkNY',
'550Northwest5thStreet',
'550ParkCenterDr',
'5555FellowshipLn',
'5580HarvestHillRd',
'55NewOxfordSt',
'5700SCicero',
'5700SLakeShoreDr',
'5706RichmondAve.',
'5729SeminoleWay',
'57AGainsfordStreet',
'57BelsizePark',
'5800SturgeonDr',
'5801SecurityBlvd',
'5830SPostRd',
'58OldSt',
'5956SherryLane',
'5BarrackRoad',
'5CenterBoulevard',
'5CurzonStreet',
'5WEBBRD',
'5east19thst',
'6',
'600QueensQuayW,Suite103,toronto',
'600necoloradost',
'601FStNW,Washington,DC',
'6039PassyunkAve',
'60McArthurRoad',
'6100NorthCharlesStreet',
'6111ESkellyDr',
'613ConstitutionDr',
'6201oldYorkRd',
'620530THAVE',
'6250HollywoodBoulevard',
'62CenterSt',
'6355MetrowestBlvd',
'635West42ndStreet',
'6380FallsviewBlvd',
'63GansevoortSt',
'641DStreetNW',
'6423PrestonshireLn',
'648Calle24',
'64Mainstreet',
'654Loria',
'658ChiswickHighRoad',
'663NewSouthPromenade',
'666GreenwichStreet',
'6701COLLINSAVE',
'6740CorbinAve',
'677PuntRd',
'6787WilsonBoulevard',
'67OldGateRoad',
'680NorthLakeShoreDrive',
'69',
'697NorthMiamiAvenue',
'6th&ChestnutStreets',
'700LafayetteSt',
'700WWashingtonSt',
'701W135thSt',
'701WOceanBlvd',
'701West135thStreet',
'70BroadSt',
'70Broadstreet',
'71-79',
'7111WCommercialBlvd',
'717WashingtonAvenue',
'7200PinesBoulevard',
'721NW9thAve.',
'721WNewEnglandAve',
'723S.BrazosSt',
'724NostrandAve',
'7333EastIndianPlaza',
'7477145thStW',
'750SBroadway',
'766EastAvenue',
'7720LakesideWoodsDr',
'777EPrincetonSt',
"777Harrah'sBlvd",
'77DunlopSt',
'78-128EhukaiSt',
'7951AlbionAvenue',
'7995GeorgiaAvenu',
'7LilacSt',
'800SpringStreetNorthwest',
'80West3rdStreet',
'8102MontereyRoad',
'82PETERST',
'8320BrookvilleRoad',
'8358PINESBLVD',
'8405WilcrestDr',
'840MariettaSt',
'8430WSunsetBlvd',
'85LafayetteRd',
'871DubuqueAvenue',
'871OldKentRd',
'8791MorganCreekLane',
'883ESanCarlosAve',
'8840CowentonAve',
'8888Southwest136th',
'8THANDCANAL',
'9000BayHillBoulevard',
'900EASTPRINCETONST.',
'9101InternationalDr',
'911WashingtonAvenue',
'915broadwayat21thstreet',
'9191OrangeDrive',
'923-927OldhamRoad',
'923E.3rd.St.',
'926HowardSt',
'932EdgewoodAve.S.',
'93Bergstrasse',
'93S.MainStreet',
'94WestHoustonStreet',
'957EastJohnBeersRoad',
'95UNIONSTREE',
'962OgdenAvenue',
'967CommonwealthAvenue',
'96LafayetteStreet,NewYorkNY',
'970GreenSt',
'9862ndAvenue',
'9939UniversalBlvd.',
'9939UniversalBoulevard',
'998AmsterdamAvenue',
'99KensingtonHighSt',
':2054PlainfieldAvenue',
'AbanaoStreet',
'AirportRoad',
'AldinePlace',
'Bachstrasse',
'BestWesternHotel(Airport)',
'BickleighBarracks',
'BlowTheWhistleOnBullying~ItMattersWhatWeDo',
'Boardwalk',
'BoylstonSt',
'BreakfastRoad',
'BroadSanctuary',
'Broadway',
'BunhillRow',
'BurnsStatueSquare,Ayr,',
'CALICECOURT',
'CATFORD',
'CLUBVALENTINO',
'Chamblee',
'CharltonRd',
'ChesfordCottages',
'ChurchHill',
'CityCampRd',
'ClockhousePlace',
'ClubPureNYC-ClubPureNY',
'ColstonStreet',
'ColumbusAvenue',
'Coursdu7emeart',
'DaddySavageRoad',
'DalstonLn',
'DavidsonHouse',
'DockRd',
'DockRoad',
'EastFairmountPark',
'EastLasOlasBlvd',
'EnergyLondonEye',
'FillmoreSt',
'FishermeadBoulevard',
'FontBLVDandLakeMerced',
'FultonStreetatSouthStreet,',
'Gateway',
'GloverStreet',
'GoldhangerRoad',
'GovernorPackRd',
'HabberleyRoadBewdley',
'HalsemaHwy',
'HammersmithRoad',
'HarneySt',
'Heisterbergalle7',
'Heisterbergallee',
'HighSt',
'HighStreet',
'HitchinRdLuton',
'HitchinRoad,LutonBedfordshire',
'HolidayInnSelect',
'HollowayRd',
'Hospitalave',
'HudlowRd',
'JackLondonSquare',
'JalanLegian',
'JalanPerintis',
'JalanRayaSukowati',
'KingstonLn',
'LeDuan',
'LeeAvenue',
'Leicester',
'LemonAve',
'LeonardWoodLoop',
'LondonOlympicStadiumLondonUnitedKingdom',
'LondonRd',
'LoxleyRd',
'MANCHESTER',
'MagsaysayAve',
'MainStreet',
'MamaNginaStreet',
'MarinaByblosHotel',
'Marr',
'MaysMeadow',
'MiddleAston',
'MidsummerBlvd',
'MiltonRoad',
'MinshullHouse,2ndFloor',
'MorrisonHillCircle',
'NA',
'NearWhiteRiver',
'NewHestonRoad',
'NewOrleansSt',
'NewSt',
'NewYorkAve',
'NewportRd',
'NorthCircusStreet',
'NorthDevonLeisureCentre',
'NorthRoad',
'OakfieldPlayingFields,',
'ParkRd',
'PeninsulaSquare',
'Pier17,SouthStreetSeaport',
'Pier40,NewYorkNY',
'PineSt',
'Pinehurst',
'PinwallLaneSheepyMagna',
'PreahMonivongBoulevard',
'PrenzlauerAllee80',
'PrincesSt',
'ProspectHillRoad',
'RanelaghGardens',
'Route739',
'SBeach',
'STRATFORDROAD,',
'SagarStreet',
'SalamisHotel,MagusaKibris',
'SamNujomaAve',
'ScarsdalePlace',
'Shortlands',
'SpencerStreet',
"StCatherine'sRetailPark",
"StMary'sRd",
'StadiumWayWest',
'Steinstreet',
'StoneyStreet',
'SugarloafPkwy',
'Suite110',
'SunsetBlvd',
'SuttonHeights,Ironbridge,Telford,Shropshire,TF74DT',
'THEBAR',
'TelephoneAvenue',
'TheClerkenwellWorkshops',
'TheHyde',
'TheMansionHouse,',
'TheSett',
"Theobald'sLn",
'ThomasStreet',
'ThunderRoad',
'ValentinaVodnika',
'Verizoncenter',
'VincentStreet',
'WALES',
'WashingtonStreet',
'West42ndStreet',
'Westway',
'WhiskeyBlue(outside),TheWHotel',
'WhitePlainsAve',
'Woodhouse',
'WrittleRd',
'WrottesleyParkRoad',
'YborSt',
'birminghamm6jct7',
'milllane',
'schlagenkamp18',
'victory park'])

bad_user_type = set([0,1,2,3,4,5])

class FraudModel(object):
    """
    Preprocess dataset from e-commerce website
    Determine the best features to keep and engineer
    Build several models to use to detect fraudulent activity
    """
    def check_against_list(self,data,lst):
        out = []
        for d in data:
            if d in lst:
                out.append(1)
            else:
                out.append(0)
        return out

    def convert_json(self):
        """
        Input: JSON Document
        Output: Dataframe

        Read in JSON document as Dataframe
        """
        self.df = pd.read_json('data/train_new.json')

    def preprocess_data(self):
        """
        Input: Dataframe
        Output: Dataframe

        Preprocessing steps:
        1. converted all event times to datetime variables and took the difference as features
        2. set all account types under fraud to be fraud for prediction
        3. converted email domains to categorical variable to use as features

        4. created boolean for images detected in description
        5. created feature to detect percent of company names in capitalization
        6. created feature to detect name length less than 2
        7. created variable to see if event has "gts" (grand total sales?)
        8. created feature to detect exclamation marks in text description
        9. created several features to detect if text description has the following: bar, city, club, dj, event, nyc, open, party, place, contact, group, life, registration, session, social, training, work, workshop

        10. created booleans to detect if event has organization description, facebook page, twitter account
        11. created feature to detect if description has url
        12. created boolean to detect if event has previously paid out
        13. created feature to detect if organization name is under the umbrella of a fraudulent list of companies

        14. extracted information on number of tickets for the event
        15. extracted information on total number of tickets for sale
        16. extracted information on total ticket costs
        17. extracted information on number of tickets sold

        18. dummified several categorical variables: 'country','currency', 'payout_type','has_header','user_type'
        19. dropped unimportant features: 'acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','email_fraud_NA'
        """

        """
        Model 1: Features: ['payee_name','venue_country_US','venue_country_PH','previous_payouts','payout_type','venue_address','email_domain','event_pubtostart','isfraud']
        """
        # self.df['payee_name_'] = self.df['payee_name'].map(lambda x: x.replace(' ', '')).apply(lambda x: 1 if x == '' else 0)

        self.df['payee_name_fruad'] = self.check_against_list(self.df['payee_name'].map(lambda x: x.replace(' ', '')).tolist(),bad_payee_name)

        self.df['domain_fruad'] = self.check_against_list(self.df['email_domain'],bad_domain)

        self.df.venue_state = [ unidecode(t) if t != None else 'UKN' for t in self.df.venue_state]

        self.df['venue_state_fruad'] = self.check_against_list(self.df['venue_state'].map(lambda x: x.replace(' ', '')).tolist(),bad_venue_state)

        self.df.venue_country = [ unidecode(t) if t != None else 'UKN' for t in self.df.venue_country]

        self.df['venue_country_fruad'] = self.check_against_list(self.df['venue_country'].map(lambda x: x.replace(' ', '')).tolist(),bad_venue_country)

        self.df['venue_address'] = self.df['venue_address'].map(lambda x: x.replace(' ', ''))

        self.df['venue_address_fraud'] = self.check_against_list(self.df['venue_address'].tolist(),bad_venue_address)

        self.df['user_type_fruad'] = self.df.user_type.isin(bad_user_type)
        # self.df['venue_country_US'] = self.df['venue_country'].apply(lambda x: 1 if x == 'US' else 0)
        # self.df['venue_country_PH'] = self.df['venue_country'].apply(lambda x: 1 if x == 'PH' else 0)
        # self.df['previous_payouts'] = self.df['previous_payouts'].apply(lambda x: 0 if not x else 1)
        # self.df['payout_type'] = self.df['payout_type'].apply(lambda x: 1 if x == 'CHECK' else 0)
        # self.df['venue_address'] = self.df['venue_address'].map(lambda x: x.replace(' ', '')).apply(lambda x: 1 if x == 'ValentinaVodnika' else 0)
        # self.df['email_domain'] = self.df['email_domain'].apply(lambda x: 1 if x == 'gmail.com' else 0)



        events = ['event_created','event_end','event_published','event_start','approx_payout_date']
        for event in events:
            self.df[event] = pd.to_datetime(self.df[event], unit='s')
        # self.df['event_duration'] = (self.df['event_end'] - self.df['event_start']).dt.days
        # self.df['event_pubtostart'] = (self.df['event_start'] - self.df['event_published']).dt.days
        self.df['event_pubtostart'] = (self.df['event_start'] - self.df['event_published']).dt.days
        fraud_accts = ['fraudster_event','fraudster','locked','tos_lock','fraudster_att']
        self.df['isfraud'] = self.df['acct_type'].apply(lambda x: x in fraud_accts)
        # self.df['country'] = self.df['country'].replace("","No Entry").fillna('No Entry')
        # self.df['listed'] = self.df.listed.apply(lambda x: x== 'y')
        # self.df['email_fraud'] = zip(self.df.email_domain.tolist(),self.df.isfraud.tolist())
        # self.df['email_fraud'] = self.df['email_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])
        # self.df['payee_fraud'] = zip(self.df['payee_name'].tolist(),self.df.isfraud.tolist())
        # self.df['payee_fraud'] = self.df['payee_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])


        '''
        Jennifer
        '''
        self.df['nogts'] = [r == 0 for r in self.df['gts']]
        self.df['has_img'] = ['img' in r for r in self.df['description']]

        self.df['name'].replace('','NO NAME GIVEN', inplace=True)
        self.df['percent_caps']=[sum(1 for c in r if c.isupper())/float(len(r)) for r in self.df['name']]
        self.df['name_lessthantwo'] = [r < 2 for r in self.df['name_length']]
        self.df['text_desc'] = [BeautifulSoup(desc, "lxml").get_text() for desc in self.df['description']]
        self.df['percent_exc'] = [sum(1 for c in r if c == '!')/float(sum(1 for c in r if c in ['.','?','!'])+1) for r in self.df['text_desc']]
        self.df['has_bar'] = ['bar' in r for r in self.df['text_desc']]
        self.df['has_city'] = ['city' in r for r in self.df['text_desc']]
        self.df['has_club'] = ['club' in r for r in self.df['text_desc']]
        self.df['has_dj'] = ['dj' in r for r in self.df['text_desc']]
        self.df['has_events'] = ['events' in r for r in self.df['text_desc']]
        self.df['has_live'] = ['live' in r for r in self.df['text_desc']]
        self.df['has_nyc'] = ['nyc' in r for r in self.df['text_desc']]
        self.df['has_open'] = ['open' in r for r in self.df['text_desc']]
        self.df['has_party'] = ['party' in r for r in self.df['text_desc']]
        self.df['has_place'] = ['place' in r for r in self.df['text_desc']]
        self.df['has_contact'] = ['contact' in r for r in self.df['text_desc']]
        self.df['has_group'] = ['group' in r for r in self.df['text_desc']]
        self.df['has_life'] = ['life' in r for r in self.df['text_desc']]
        self.df['has_registration'] = ['registration' in r for r in self.df['text_desc']]
        self.df['has_session'] = ['session' in r for r in self.df['text_desc']]
        self.df['has_social'] = ['social' in r for r in self.df['text_desc']]
        self.df['has_training'] = ['training' in r for r in self.df['text_desc']]
        self.df['has_work'] = ['work' in r for r in self.df['text_desc']]
        self.df['has_workshop'] = ['workshop' in r for r in self.df['text_desc']]

        '''
        Jesse
        '''
        self.df['has_orgdesc'] = self.df.org_desc.apply(lambda x: x != '')
        self.df['has_fbkcateg'] = self.df.org_facebook.apply(lambda x: x != 0)
        self.df['has_twitteracctnum'] = self.df.org_twitter.apply(lambda x: x != 0)
        self.df['has_url'] = ['http' in r for r in self.df['org_desc']]
        self.df['payout_type'] = self.df['payout_type'].replace('','UNK')
        self.df['has_prev_payout'] = self.df['previous_payouts'].apply(lambda x: 0 if not x else 1)
        fraud_list = ['LIself.df','Global Gas Card','Ultimate Wine','Pocket Pictures', 'FORD MODELS UK',
                  'Rotary Club of East Los Angeles', 'Tree of Life', 'Startup Saturdays', 'Gametightny.com', 'Ger-Nis Culinary & Herb Center',
                  'Joonbug', 'Market District Robinson', 'Premier Events', 'Pocket Pictures', 'STYLEPARTIES', 'Blow The Whistle On Bullying ~ It Matters What We do',
                  'Network After Work','Museum of Contemporary Art, North Miami', "Mabs' Events", 'DC Black Theatre Festival', 'stephen',
                  '1st Class Travel Club', 'ELITE SOCIAL', 'Absolution Chess Club']
        self.df['org_name_naughty_list'] = self.df['org_name'].isin(fraud_list)

        '''
        Muneeb
        '''
        fs = [self.get_num_of_tickets, self.get_total_number_of_tickets_for_sale, self.get_total_ticket_costs, self.get_number_of_tickets_sold] #
        ns = ['num_of_tickets', 'total_number_of_tickets_for_sale', 'total_ticket_costs', 'number_of_tickets_sold'] #
        for i,f in enumerate(fs):
            self.df[ns[i]] = self.df['ticket_types'].map(f)


        # self.df = pd.get_dummies(self.df, columns=['country','currency', 'payout_type','has_header','user_type','email_fraud']) #
        # dropped_columns = ['acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','email_fraud_NA'] #
        # self.df_rf = self.df.drop(dropped_columns, axis=1)
        # keep = ['payee_name_fruad','event_pubtostart','isfraud','num_of_tickets', 'total_number_of_tickets_for_sale', 'total_ticket_costs', 'number_of_tickets_sold','sale_duration','nogts','domain_fruad','venue_state_fruad','venue_country_fruad','venue_address_fraud','has_img',]
        dropped_columns = ['acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','currency','country','payout_type','user_type']
        self.df = self.df.drop(dropped_columns,axis=1)
        return self.df

    def get_num_of_tickets(self, ticket_types):
        """
        Output: Int
        """
        return len(ticket_types)

    def get_total_number_of_tickets_for_sale(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_total']
        return total

    def get_total_ticket_costs(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_total'] * d['cost']
        return total

    def get_number_of_tickets_sold(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_sold']
        return total


    def perform_grid_search_rf(self, X_train, X_test, y_train, y_test):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """

        random_forest_grid = {'n_estimators': [100],
                                'criterion': ['gini','entropy'],
                                'min_samples_split': [5],
                                'min_samples_leaf': [1,2,3,4],
                                'n_jobs': [-1],
                                'max_features': [None],
                                'class_weight': ['balanced',None,'balanced_subsample']}

        rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                     random_forest_grid,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='accuracy',
                                     cv=5)

        rf_gridsearch.fit(X_train, y_train)

        print "best parameters:", rf_gridsearch.best_params_

        best_rf_model = rf_gridsearch.best_estimator_

        y_pred = best_rf_model.predict(X_test)

        print "Accuracy with best rf:", cross_val_score(best_rf_model, X_test, y_test, scoring='accuracy').mean()

        rf = RandomForestClassifier(n_estimators=10, oob_score=True, class_weight='balanced')

        print "Accuracy with default param rf:", cross_val_score(rf, X_test, y_test, scoring='accuracy').mean()

        return best_rf_model, y_pred


    def plot_confusion_matrix(self, y_test, y_pred, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        Output: Confusion Matrix Plot

        Plot confusion matrix to determine the number of false positive and false negatives produced from our model
        """
        cm = self.create_confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print 'Confusion matrix, without normalization'
        print cm
        plt.clf()
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.grid([])
        plt.savefig('confusion_mat_fraud.png')

    def create_confusion_matrix(self, y_test, y_pred):
        """
        Output: Confusion matrix with appropriate labels
        """
        return pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True']).reindex_axis([1,0], axis=1).reindex_axis([1,0], axis=0)

    def print_featimpt(self, df2, best_rf_model, percent=.99):
        """
        Output: Feature Importances from Random Forest
        """
        lab_feats = sorted(zip(df2.columns, best_rf_model.feature_importances_), key=lambda x : x[1])[::-1]

        total,cnt = 0,0
        for n,v in lab_feats:
            total+=v
            if total<=percent:
                cnt+=1
                print cnt,n,v

    def pickle_model(self, model, name):
        """
        Output: Saved Model

        Pickles our model for later use
        """
        with open("fraud_app/data/{}.pkl".format(name), 'w') as f:
            pickle.dump(model, f)


def make_pred_prob_plot_data(model, df, column):
    dfc = df.copy() 
    rng = np.linspace(df[column].min(), df[column].max())
    probs = []
    for val in rng:
        dfc[column] = val
        pred_probs = model.predict_proba(dfc)[:, 1]
        probs.append([boot_sample.mean() for boot_sample in (resample(pred_probs) for _ in xrange(1000))])
    return rng, np.array(probs).T



def probability_plot(model, df, column, fname):  #'fname' is filename; 'column' is the string column name of the feature to graph 
    rng, probs = make_pred_prob_plot_data(model, df, column)
    fig, ax1 = plt.subplots()
    prob_means = probs.mean(axis=0)
    upper_bounds = np.percentile(probs, q=95, axis=0)
    lower_bounds = np.percentile(probs, q=5, axis=0)
    ax1.plot(rng, prob_means, color='k')
    ax1.fill_between(rng, lower_bounds, upper_bounds, facecolor='k', alpha=0.25)
    ax1.set_xlabel(column, labelpad=11)
    ax1.set_ylabel('Predicted Probability', labelpad=11)
    ax2 = ax1.twinx()
    ax2.hist(df[column].values, color='k', alpha=0.15)
    ax2.set_ylabel('Frequency', labelpad=11)
    plt.tight_layout() 

    plt.savefig(fname)
    plt.close()             


if __name__ == '__main__':
    fm = FraudModel()
    fm.convert_json()
    df2 = fm.preprocess_data()
    df2.dropna(inplace=True)

    y = df2.pop('isfraud').values
    X = df2.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    best_rf_model, y_pred = fm.perform_grid_search_rf(X_train, X_test, y_train, y_test)

    fm.print_featimpt(df2, best_rf_model)

    fm.plot_confusion_matrix(y_test, y_pred, [1,0])

    # df2.to_csv("fraud_app/data/fraud_data.csv", index=False)

    # fm.pickle_model(best_rf_model, name='model5_small')

    # df3 = pd.concat([pd.DataFrame(df2.columns.tolist(), columns=['columns']), pd.DataFrame(best_rf_model.feature_importances_.tolist(), columns=['featimpt'])], axis=1).sort_values(by='featimpt', ascending=False)
    #
    # p = Bar(df3, 'columns', values='featimpt', title="Feature Importance From Model")
    #
    # output_file("feat_impt.html", title="Feature Importance From Model")
    #
    # show(p)
