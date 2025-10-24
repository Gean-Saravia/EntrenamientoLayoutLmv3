Base de Datos - Facturas_db

Codigo para la creacion de la tabla para guardar el resultados de las imagenes:
CREATE TABLE [dbo].[datos_imagenes] (
    [id] INT IDENTITY(1,1) PRIMARY KEY,
    [nombre_imagen] NVARCHAR(255) NULL,
    [n_cliente] NVARCHAR(100) NULL,
    [nombre] NVARCHAR(100) NULL,
    [apellido] NVARCHAR(100) NULL,
    [email] NVARCHAR(255) NULL,
    [fecha_factura] DATE NULL,
    [monto_total] DECIMAL(18,2) NULL,
    [fecha_subida] DATETIME NULL DEFAULT GETDATE()
);
Aclaracion: Esta tabla permite nulos debido a que el modelo esta entrenado con poca precision, debido a que aveces no reconoce campos este mismo puede llegar a guardar nulos, para evitar la perdida de datos y verificar el entrenamiento se decidio permitir nulos
, a futuro esta tabla estará formada de forma más rigurosa para aumentar el nivel de nuestro modelo.
