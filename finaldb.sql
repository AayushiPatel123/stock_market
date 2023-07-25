/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - stockmarket
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`stockmarket` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `stockmarket`;

/*Table structure for table `tblregister` */

DROP TABLE IF EXISTS `tblregister`;

CREATE TABLE `tblregister` (
  `id` int(255) NOT NULL auto_increment,
  `uname` varchar(255) default NULL,
  `email` varchar(255) default NULL,
  `password` varchar(255) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `tblregister` */

insert  into `tblregister`(`id`,`uname`,`email`,`password`) values (1,'roshan','a@gmail.com','r'),(2,'amol','amol@gmail.com','amol');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
