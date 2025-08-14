string SendPredictionRequest(const double &prices[])
  {
   string json = "{\"prices\":[";

   for(int i=0; i<ArraySize(prices); i++)
     {
      json += DoubleToString(prices[i], 5);
      if(i < ArraySize(prices) - 1) json += ",";
     }
   json += "]}";

   char result[];
   string headers;
   string url = "http://127.0.0.1:5000/predict";

   char post[];
   StringToCharArray(json, post);

   if(WebRequest("POST", url, "application/json", headers, post, result, 5000))
     return CharArrayToString(result);
   else
     Print("WebRequest failed: ", GetLastError());

   return "";
  }