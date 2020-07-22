package org.cwinteractive.photon;

import io.quarkus.runtime.QuarkusApplication;


import io.quarkus.picocli.runtime.annotations.TopCommand;
import picocli.CommandLine;


@TopCommand
@CommandLine.Command(mixinStandardHelpOptions = true, subcommands =  {TYTrainingModule.class, TYTestingModule.class})
public class PhotonMain  { }